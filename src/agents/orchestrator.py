"""
Agent orchestrator — coordinates the query routing and agent execution chain.
"""

from typing import Dict, Any, Optional
import time
from datetime import datetime
from src.common import get_agent_logger, AgentExecutionError
from .agents import QueryRouter, InformationAgent, PlanningAgent, RecommendationAgent, XAIValidator


class AgentOrchestrator:
    """Coordinates execution of the agent chain for each user query."""

    def __init__(self, llm, tools: Optional[Dict[str, Any]] = None, enable_rag: bool = True):
        self.logger = get_agent_logger("Orchestrator")
        self.llm = llm
        self.tools = tools or {}
        self.enable_rag = enable_rag

        self.query_router = QueryRouter(llm)
        self.info_agent = InformationAgent(llm, tools)
        self.planning_agent = PlanningAgent(llm, tools)
        self.recommendation_agent = RecommendationAgent(llm, tools)
        self.xai_validator = XAIValidator(llm)

        self.logger.info("Agent orchestrator initialized")

    def execute(self, query: str, user_context: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
        try:
            start_time = time.time()
            self.logger.info(f"Executing query: {query}")

            # Step 1: Route
            routing = self.query_router.execute(query=query)
            query_type = routing["query_type"]
            # Router extraction takes priority; fall back to anything passed in context
            location = routing.get("location") or (user_context or {}).get("destination")
            origin = routing.get("origin") or (user_context or {}).get("origin_city")
            tools_needed = routing.get("tools_needed", [])
            # Propagate resolved location back into routing so UI can display it
            routing["location"] = location

            agent_outputs = {"query_router": routing}

            # Step 2: Non-travel check
            if not routing.get("is_travel_related", True):
                response = self.llm.generate(
                    f'The user asked: "{query}". This does not appear to be a travel question. '
                    f"Politely explain that you specialise in travel planning and ask what travel help they need.",
                    system_message="You are a travel assistant.",
                    max_tokens=150,
                )
                return {
                    "query": query,
                    "response": response,
                    "agent_trace": agent_outputs,
                    "execution_time_ms": (time.time() - start_time) * 1000,
                    "validation": {"confidence_score": 0.95, "validation_status": "approved"},
                    "timestamp": datetime.utcnow().isoformat(),
                }

            # Step 3: Execute the right agent
            if query_type == "PLANNING":
                try:
                    budget = (user_context or {}).get("budget") or kwargs.get("budget")
                    result = self.planning_agent.execute(
                        query=query,
                        location=location,
                        origin=origin,
                        budget=float(budget) if budget else None,
                        start_date=(user_context or {}).get("start_date"),
                        end_date=(user_context or {}).get("end_date"),
                        preferences=(user_context or {}).get("travel_preferences"),
                    )
                    agent_outputs["planning_agent"] = result
                except Exception as e:
                    self.logger.error(f"PlanningAgent failed: {e}", exc_info=True)
                    agent_outputs["planning_agent"] = {"error": str(e)}

            elif query_type == "RECOMMENDATION":
                try:
                    result = self.recommendation_agent.execute(
                        query=query,
                        location=location,
                        tools_needed=tools_needed,
                        traveler_profile=user_context,
                    )
                    agent_outputs["recommendation_agent"] = result
                except Exception as e:
                    self.logger.error(f"RecommendationAgent failed: {e}", exc_info=True)
                    agent_outputs["recommendation_agent"] = {"error": str(e)}

            else:  # INFORMATION (default)
                try:
                    result = self.info_agent.execute(
                        query=query,
                        location=location,
                        location_country=routing.get("location_country"),
                        origin=origin,
                        origin_country=routing.get("origin_country"),
                        tools_needed=tools_needed,
                    )
                    agent_outputs["information_agent"] = result
                except Exception as e:
                    self.logger.error(f"InformationAgent failed: {e}", exc_info=True)
                    agent_outputs["information_agent"] = {"error": str(e)}

            # Step 4: Validate
            validation = self.xai_validator.execute(
                agent_outputs=agent_outputs,
                original_query=query,
            )
            agent_outputs["xai_validator"] = validation

            execution_time_ms = (time.time() - start_time) * 1000

            return {
                "query": query,
                "response": self._aggregate_response(agent_outputs),
                "agent_trace": agent_outputs,
                "execution_time_ms": execution_time_ms,
                "validation": validation,
                "timestamp": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            self.logger.error(f"Orchestration failed: {e}", exc_info=True)
            raise AgentExecutionError(f"Orchestration failed: {e}", agent_name="Orchestrator", context={"query": query})

    def _aggregate_response(self, agent_outputs: Dict[str, Any]) -> str:
        if "planning_agent" in agent_outputs:
            data = agent_outputs["planning_agent"]
            if isinstance(data, dict) and data.get("itinerary"):
                return data["itinerary"]

        if "information_agent" in agent_outputs:
            data = agent_outputs["information_agent"]
            if isinstance(data, dict) and data.get("answer"):
                return data["answer"]

        if "recommendation_agent" in agent_outputs:
            data = agent_outputs["recommendation_agent"]
            if isinstance(data, dict) and data.get("recommendations"):
                return data["recommendations"]

        return "I was unable to process your query. Please try rephrasing it."

    def get_agent_status(self) -> Dict[str, str]:
        return {
            "query_router": "ready",
            "information": "ready",
            "planning": "ready",
            "recommendation": "ready",
            "xai_validator": "ready",
        }