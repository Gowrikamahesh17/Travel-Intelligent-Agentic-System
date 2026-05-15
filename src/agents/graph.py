"""
LangGraph-based orchestration for the Travel Intelligent Agentic System.

GRAPH ARCHITECTURE
------------------
                        ┌─────────────────┐
                        │   query_router   │  (classifies, extracts entities,
                        │                 │   decomposes multi-intent queries)
                        └────────┬────────┘
                                 │
              ┌──────────────────┼──────────────────┐
              │                  │                   │
        INFORMATION           PLANNING         RECOMMENDATION
              │                  │                   │
    ┌─────────▼────────┐ ┌──────▼──────┐  ┌────────▼────────┐
    │ information_agent│ │planning_    │  │recommendation_  │
    │ (runs tools per  │ │agent        │  │agent            │
    │  sub-query)      │ │             │  │                 │
    └─────────┬────────┘ └──────┬──────┘  └────────┬────────┘
              │                  │                   │
              └──────────────────┼───────────────────┘
                                 │
                        ┌────────▼────────┐
                        │  xai_validator  │  (confidence scoring, source tagging)
                        └────────┬────────┘
                                 │
                              END / response

KEY IMPROVEMENTS OVER OLD IF/ELIF ORCHESTRATOR
----------------------------------------------
1. LangGraph manages state and routing — no hardcoded if/elif chains.
2. Multi-intent queries are decomposed into sub-queries by the router,
   each processed independently and merged at the end.
3. The graph is inspectable (draw_mermaid(), get_graph()) for debugging.
4. State is a typed dict — all agents share a well-defined schema.
5. Each node is a pure function(state) → state_update (easy to test in isolation).
"""

from __future__ import annotations

import time
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, TypedDict, Annotated, Literal
import operator

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from src.common import get_agent_logger
from .agents import (
    QueryRouter, InformationAgent, PlanningAgent,
    RecommendationAgent, XAIValidator, _GUARDRAIL_SYSTEM,
)

logger = get_agent_logger("LangGraph")


# ── Shared graph state ────────────────────────────────────────────────────────

class TravelState(TypedDict):
    """State that flows through every node in the graph."""
    # Input
    query: str
    user_context: Dict[str, Any]

    # Router outputs
    query_type: str                        # INFORMATION | PLANNING | RECOMMENDATION
    location: Optional[str]
    origin: Optional[str]
    tools_needed: List[str]
    is_travel_related: bool
    sub_queries: List[Dict[str, Any]]      # For multi-intent decomposition

    # Agent outputs (accumulated across sub-queries)
    agent_outputs: Dict[str, Any]

    # Final
    response: str
    validation: Dict[str, Any]
    execution_time_ms: float
    timestamp: str


# ── Node functions ────────────────────────────────────────────────────────────

def router_node(state: TravelState, *, agents: Dict[str, Any]) -> Dict[str, Any]:
    """
    Route the query: classify type, extract entities, detect multi-intent.
    Returns state updates only (LangGraph merges these into the full state).
    """
    qr: QueryRouter = agents["query_router"]
    query = state["query"]

    routing = qr.execute(query=query)
    query_type = routing["query_type"]
    # Router now returns clean city names separately from country names
    location = routing.get("location") or state.get("user_context", {}).get("destination")
    origin   = routing.get("origin")   or state.get("user_context", {}).get("origin_city")
    tools_needed = routing.get("tools_needed", [])

    location_country = routing.get("location_country")
    origin_country   = routing.get("origin_country")

    # Multi-intent detection: split compound queries into discrete sub-queries
    sub_queries = _decompose_query(query, query_type, location, origin, tools_needed,
                                   location_country, origin_country)

    logger.info(
        f"Router: type={query_type} city={location!r} country={location_country!r} "
        f"origin={origin!r} tools={tools_needed} sub_queries={len(sub_queries)}"
    )

    return {
        "query_type": query_type,
        "location": location,
        "origin": origin,
        "tools_needed": tools_needed,
        "is_travel_related": routing.get("is_travel_related", True),
        "sub_queries": sub_queries,
        "agent_outputs": {"query_router": routing},
    }


def _decompose_query(query: str, query_type: str, location: Optional[str],
                     origin: Optional[str], tools_needed: List[str],
                     location_country: Optional[str] = None,
                     origin_country: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Break a compound query into discrete sub-queries.
    Example: "flights from Frankfurt to Delhi AND restaurants in Delhi"
    → [{type:INFORMATION, tools:[flights], origin:Frankfurt, location:Delhi},
       {type:RECOMMENDATION, tools:[restaurants], location:Delhi}]
    """
    base = {
        "query": query, "location": location, "origin": origin,
        "location_country": location_country, "origin_country": origin_country,
    }

    # If only one concern, no decomposition needed
    if len(tools_needed) <= 1:
        return [{**base, "query_type": query_type, "tools_needed": tools_needed}]

    # Separate tools by the type of agent they need
    flight_tools = [t for t in tools_needed if t == "flights"]
    info_tools   = [t for t in tools_needed if t in ("weather", "visa", "health", "transport")]
    rec_tools    = [t for t in tools_needed if t in ("hotels", "restaurants", "cultural_info")]

    sub_queries = []
    if flight_tools:
        sub_queries.append({**base, "query_type": "INFORMATION", "tools_needed": flight_tools})
    if info_tools:
        sub_queries.append({**base, "query_type": "INFORMATION", "tools_needed": info_tools})
    if rec_tools:
        sub_queries.append({**base, "query_type": "RECOMMENDATION", "tools_needed": rec_tools})

    return sub_queries if sub_queries else [{**base, "query_type": query_type, "tools_needed": tools_needed}]


def _route_decision(state: TravelState) -> str:
    """LangGraph conditional edge: decide which node to go to next."""
    if not state.get("is_travel_related", True):
        return "non_travel"
    # If there are sub-queries, always go to the dispatcher
    if state.get("sub_queries"):
        return "dispatcher"
    return "dispatcher"


def dispatcher_node(state: TravelState, *, agents: Dict[str, Any]) -> Dict[str, Any]:
    """
    Processes all sub-queries sequentially and merges their outputs.
    Each sub-query is routed to the correct agent.
    """
    sub_queries = state.get("sub_queries", [])
    agent_outputs = dict(state.get("agent_outputs", {}))
    merged_answers = []

    info_agent: InformationAgent     = agents["information_agent"]
    plan_agent: PlanningAgent        = agents["planning_agent"]
    rec_agent:  RecommendationAgent  = agents["recommendation_agent"]
    user_ctx = state.get("user_context", {})

    # Nationality → approximate hub airport for fallback origin
    NATIONALITY_HUB = {
        "US": "New York", "UK": "London", "Germany": "Frankfurt",
        "DE": "Frankfurt", "France": "Paris", "India": "Delhi",
        "Japan": "Tokyo", "Australia": "Sydney", "Canada": "Toronto",
        "Spain": "Madrid", "Italy": "Rome", "Netherlands": "Amsterdam",
    }
    nationality = user_ctx.get("nationality", "")
    nationality_origin = NATIONALITY_HUB.get(nationality, "")

    for i, sq in enumerate(sub_queries):
        sq_type     = sq.get("query_type", "INFORMATION")
        sq_location = sq.get("location")
        sq_origin   = sq.get("origin") or nationality_origin or None
        sq_tools    = sq.get("tools_needed", [])
        sq_query    = sq.get("query", state["query"])

        logger.info(f"Dispatcher: sub-query {i+1}/{len(sub_queries)} type={sq_type} tools={sq_tools}")

        try:
            if sq_type == "PLANNING":
                budget = user_ctx.get("budget")
                result = plan_agent.execute(
                    query=sq_query,
                    location=sq_location,
                    origin=sq_origin,
                    budget=float(budget) if budget else None,
                    start_date=user_ctx.get("start_date"),
                    end_date=user_ctx.get("end_date"),
                    preferences=user_ctx.get("travel_preferences"),
                )
                key = f"planning_agent_{i}" if i > 0 else "planning_agent"
                agent_outputs[key] = result
                if result.get("itinerary"):
                    merged_answers.append(result["itinerary"])

            elif sq_type == "RECOMMENDATION":
                result = rec_agent.execute(
                    query=sq_query,
                    location=sq_location,
                    tools_needed=sq_tools,
                    traveler_profile=user_ctx,
                )
                key = f"recommendation_agent_{i}" if i > 0 else "recommendation_agent"
                agent_outputs[key] = result
                if result.get("recommendations"):
                    merged_answers.append(result["recommendations"])

            else:  # INFORMATION
                result = info_agent.execute(
                    query=sq_query,
                    location=sq_location,
                    location_country=sq.get("location_country"),
                    origin=sq_origin,
                    origin_country=sq.get("origin_country"),
                    tools_needed=sq_tools,
                )
                key = f"information_agent_{i}" if i > 0 else "information_agent"
                agent_outputs[key] = result
                if result.get("answer"):
                    merged_answers.append(result["answer"])

        except Exception as e:
            logger.error(f"Sub-query {i+1} failed: {e}", exc_info=True)
            agent_outputs[f"error_{i}"] = {"error": str(e)}

    # Combine all answers
    combined = "\n\n---\n\n".join(merged_answers) if merged_answers else ""

    return {
        "agent_outputs": agent_outputs,
        "response": combined,
    }


def non_travel_node(state: TravelState, *, agents: Dict[str, Any]) -> Dict[str, Any]:
    """Handle non-travel queries with a polite redirect."""
    llm = agents["llm"]
    response = llm.generate(
        f'The user asked: "{state["query"]}". '
        f"This is not a travel question. "
        f"Politely explain you specialise in travel and suggest what you can help with.",
        system_message=_GUARDRAIL_SYSTEM,
        max_tokens=150,
    )
    return {
        "response": response,
        "agent_outputs": dict(state.get("agent_outputs", {})),
    }


def validator_node(state: TravelState, *, agents: Dict[str, Any]) -> Dict[str, Any]:
    """XAI validation — computes confidence from data provenance, no LLM call."""
    xai: XAIValidator = agents["xai_validator"]
    agent_outputs = state.get("agent_outputs", {})
    validation = xai.execute(agent_outputs=agent_outputs, original_query=state["query"])

    # Attach data-source summary to the response
    response = state.get("response", "")
    sources = _summarise_sources(agent_outputs)
    if sources and response:
        response = response + f"\n\n---\n*Data sources: {sources}*"

    return {
        "validation": validation,
        "response": response,
        "timestamp": datetime.utcnow().isoformat(),
    }


def _summarise_sources(agent_outputs: Dict[str, Any]) -> str:
    """Build a one-line source summary from all tool_data in the outputs."""
    sources = set()
    for key, val in agent_outputs.items():
        if not isinstance(val, dict):
            continue
        for tool_name, tool_result in val.get("tool_data", {}).items():
            if isinstance(tool_result, dict):
                src = tool_result.get("source", "")
                if src and "llm_knowledge" not in src:
                    sources.add(src.split("(")[0].strip())
    return ", ".join(sorted(sources)) if sources else ""


# ── Graph builder ─────────────────────────────────────────────────────────────

def build_travel_graph(llm: Any, tools: Dict[str, Any]) -> StateGraph:
    """
    Build and compile the LangGraph state machine.

    Nodes:
      query_router  → classifies, extracts, decomposes
      dispatcher    → runs sub-queries through the correct agents
      non_travel    → handles off-topic queries
      xai_validator → confidence scoring

    Edges:
      query_router --(travel)--> dispatcher
      query_router --(non-travel)--> non_travel
      dispatcher   --> xai_validator
      non_travel   --> xai_validator
      xai_validator --> END
    """
    # Instantiate agents
    agent_map = {
        "llm": llm,
        "query_router":         QueryRouter(llm),
        "information_agent":    InformationAgent(llm, tools),
        "planning_agent":       PlanningAgent(llm, tools),
        "recommendation_agent": RecommendationAgent(llm, tools),
        "xai_validator":        XAIValidator(llm),
    }

    # Wrap node functions with the agent_map as closure
    def _router(state):     return router_node(state, agents=agent_map)
    def _dispatch(state):   return dispatcher_node(state, agents=agent_map)
    def _non_travel(state): return non_travel_node(state, agents=agent_map)
    def _validate(state):   return validator_node(state, agents=agent_map)

    # Build the graph
    graph = StateGraph(TravelState)

    graph.add_node("query_router",  _router)
    graph.add_node("dispatcher",    _dispatch)
    graph.add_node("non_travel",    _non_travel)
    graph.add_node("xai_validator", _validate)

    # Entry point
    graph.set_entry_point("query_router")

    # Conditional edge from router
    graph.add_conditional_edges(
        "query_router",
        _route_decision,
        {
            "dispatcher":  "dispatcher",
            "non_travel":  "non_travel",
        },
    )

    # Linear edges after routing
    graph.add_edge("dispatcher",  "xai_validator")
    graph.add_edge("non_travel",  "xai_validator")
    graph.add_edge("xai_validator", END)

    return graph.compile()


# ── LangGraph-backed Orchestrator ─────────────────────────────────────────────

class LangGraphOrchestrator:
    """
    Drop-in replacement for AgentOrchestrator that uses LangGraph internally.
    The public interface (execute, get_agent_status) is identical.
    """

    def __init__(self, llm: Any, tools: Optional[Dict[str, Any]] = None,
                 enable_rag: bool = True):
        self.logger = get_agent_logger("LangGraphOrchestrator")
        self.llm = llm
        self.tools = tools or {}
        self.compiled_graph = build_travel_graph(llm, self.tools)
        self.logger.info("LangGraph orchestrator initialised")

    def execute(self, query: str,
                user_context: Optional[Dict[str, Any]] = None,
                **kwargs) -> Dict[str, Any]:
        """Run the query through the LangGraph state machine."""
        start_time = time.time()
        self.logger.info(f"LangGraph executing: {query}")

        # Initial state
        initial_state: TravelState = {
            "query": query,
            "user_context": user_context or {},
            "query_type": "INFORMATION",
            "location": None,
            "origin": None,
            "tools_needed": [],
            "is_travel_related": True,
            "sub_queries": [],
            "agent_outputs": {},
            "response": "",
            "validation": {},
            "execution_time_ms": 0.0,
            "timestamp": "",
        }

        # Run the graph
        final_state = self.compiled_graph.invoke(initial_state)

        execution_time_ms = (time.time() - start_time) * 1000
        self.logger.info(f"LangGraph completed in {execution_time_ms:.0f}ms")

        return {
            "query": query,
            "response": final_state.get("response", ""),
            "agent_trace": final_state.get("agent_outputs", {}),
            "execution_time_ms": execution_time_ms,
            "validation": final_state.get("validation", {}),
            "timestamp": final_state.get("timestamp", datetime.utcnow().isoformat()),
        }

    def get_agent_status(self) -> Dict[str, str]:
        return {
            "query_router": "ready",
            "information_agent": "ready",
            "planning_agent": "ready",
            "recommendation_agent": "ready",
            "xai_validator": "ready",
            "orchestrator": "langgraph",
        }

    def get_graph_diagram(self) -> str:
        """Return Mermaid diagram of the graph for documentation."""
        try:
            return self.compiled_graph.get_graph().draw_mermaid()
        except Exception:
            return "graph diagram unavailable"
