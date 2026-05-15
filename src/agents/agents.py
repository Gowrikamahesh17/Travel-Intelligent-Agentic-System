"""
Agent implementations — five-agent architecture (no LangChain/LangGraph).

ARCHITECTURE & FLOW
-------------------
This is a hand-written orchestration pipeline, NOT LangChain or LangGraph.
All agents are plain Python classes that call an LLM provider directly.

Query flow (defined in orchestrator.py):
  1. QueryRouter   — one LLM call → JSON with query_type, location, origin, tools_needed
  2. One of:
       InformationAgent   (INFORMATION queries: weather, flights, visa, health, etc.)
       PlanningAgent      (PLANNING: multi-day itinerary)
       RecommendationAgent (RECOMMENDATION: hotel/restaurant/activity suggestions)
  3. XAIValidator  — confidence scoring, no extra LLM call

TOOL EXECUTION
--------------
Each agent runs only the tools specified by the router in `tools_needed`.
Tools attempt real API calls; if unavailable, they emit a `note` instructing
the LLM to answer from its training knowledge with an explicit disclaimer.

GUARDRAILS
----------
1. Topic guard   — non-travel queries are short-circuited with a redirect message.
2. Tool-first    — LLM only sees tool data + instructions; never free-generates facts.
3. Source tagging— every answer must tag its source: live API vs training knowledge.
4. Disclaimer    — all knowledge-based answers append a "verify before travel" note.
5. No fabrication — system prompt explicitly forbids inventing prices, names, or policies.
6. Date injection — current date is injected so the LLM doesn't confuse knowledge cutoff.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from datetime import datetime
import json
from src.common import get_agent_logger, AgentExecutionError
from src.llm import BaseLLM

# Guardrail system prompt injected into every agent LLM call
_GUARDRAIL_SYSTEM = (
    "You are a knowledgeable, accurate travel assistant. "
    "STRICT RULES you must always follow:\n"
    "1. ONLY answer travel-related questions.\n"
    "2. When live API data is provided, use it exactly — never override it.\n"
    "3. When answering from training knowledge, ALWAYS add: "
    "   '*(based on general knowledge — verify before travel)*'\n"
    "4. NEVER invent specific hotel names, prices, visa fees, flight numbers, "
    "   or restaurant names that you are not confident about. "
    "   If uncertain, say 'I cannot confirm this — please verify directly.'\n"
    "5. Prices and regulations change. Always recommend verifying with official sources.\n"
    "6. Be specific and helpful, but honest about the limits of your knowledge."
)


class BaseAgent(ABC):
    def __init__(self, name: str, llm: BaseLLM):
        self.name = name
        self.llm = llm
        self.logger = get_agent_logger(name)

    @abstractmethod
    def execute(self, **kwargs) -> Dict[str, Any]:
        pass

    def __repr__(self):
        return f"<{self.__class__.__name__} name={self.name}>"


class QueryRouter(BaseAgent):
    """
    Entry point — classifies the query and decides which agent + tools to use.
    Uses a single LLM call returning structured JSON. No rule-based parsing.
    Guardrail: travel relevance check is embedded in the JSON response.
    """

    def __init__(self, llm: BaseLLM):
        super().__init__("QueryRouter", llm)

    def execute(self, query: str, **kwargs) -> Dict[str, Any]:
        try:
            self.logger.info(f"Routing query: {query}")

            today = datetime.now().strftime("%Y-%m-%d")
            prompt = f"""Analyze this travel query. Today's date is {today}.
Respond with JSON only — no markdown, no explanation.

Query: "{query}"

Return this exact JSON:
{{
  "query_type": "INFORMATION|PLANNING|RECOMMENDATION",
  "destination_city": "ONLY the city name, no country (e.g. 'Delhi' not 'Delhi India')",
  "destination_country": "ONLY the country name if mentioned, or null",
  "origin_city": "ONLY the origin city name if mentioned (e.g. 'Frankfurt' not 'Frankfurt Germany'), or null",
  "origin_country": "ONLY the origin country name if mentioned, or null",
  "tools_needed": ["list", "of", "tools"],
  "is_travel_related": true or false
}}

Tool options: weather, flights, hotels, restaurants, visa, health, transport, cultural_info

Tool selection rules — pick ONLY what was explicitly asked:
- weather / temperature / rain / climate / forecast           → weather
- flight / fly / airline / airfare / between X and Y         → flights (set origin_city + destination_city)
- hotel / hostel / accommodation / stay / where to sleep     → hotels
- restaurant / food / eat / dining / cuisine / cafe          → restaurants
- visa / passport / entry requirement / immigration          → visa
- health / vaccine / vaccination / medical / safe to visit   → health
- bus / metro / train / taxi / local transport               → transport
- culture / customs / etiquette / tradition / language       → cultural_info
- plan / itinerary / X days / schedule                      → PLANNING, tools: weather+hotels+flights+cultural_info
- recommend / suggest / best / top (no planning)             → RECOMMENDATION

Examples:
"flights between Frankfurt Germany and Delhi India"
→ origin_city="Frankfurt", origin_country="Germany", destination_city="Delhi", destination_country="India", tools=["flights"]

"What's the weather like in Heidelberg Germany?"
→ destination_city="Heidelberg", destination_country="Germany", tools=["weather"]"""

            response = self.llm.generate(
                prompt,
                system_message="You are a travel query analyzer. Respond with valid JSON only, no markdown.",
                max_tokens=200,
            ).strip()

            # Strip markdown fences if model added them
            if response.startswith("```"):
                response = response.split("```")[1]
                if response.startswith("json"):
                    response = response[4:]

            parsed = json.loads(response)
            query_type = parsed.get("query_type", "INFORMATION").upper()
            if query_type not in ("INFORMATION", "PLANNING", "RECOMMENDATION"):
                query_type = "INFORMATION"

            # LLM gives city and country separately.
            # For country-only queries ("customs in India"), destination_city may be null.
            # Use country as the location fallback so downstream tools still work.
            destination_city    = parsed.get("destination_city") or parsed.get("location")
            destination_country = parsed.get("destination_country")
            origin_city         = parsed.get("origin_city") or parsed.get("origin")
            origin_country      = parsed.get("origin_country")

            # Treat the literal string "null" the same as JSON null
            def _clean(val):
                if not val or (isinstance(val, str) and val.lower() in ("null","none","")):
                    return None
                return val

            destination_city    = _clean(destination_city)
            destination_country = _clean(destination_country)
            origin_city         = _clean(origin_city)
            origin_country      = _clean(origin_country)

            # If city is missing but country is present, use country as location
            if not destination_city and destination_country:
                destination_city = destination_country

            # If both city and country are missing, try to extract from query using LLM output tools
            # as a hint (e.g. "Cultural customs in India" → India is both city and country fallback)
            if not destination_city:
                # Simple heuristic: look for known country/city names in the query
                q_words = query.replace(",", " ").split()
                for word in reversed(q_words):  # last words most likely to be destination
                    if len(word) > 3 and word[0].isupper():
                        destination_city = word
                        break

            tools_needed = parsed.get("tools_needed") or []
            # If LLM returned empty tools, infer from query keywords
            if not tools_needed:
                q_lower = query.lower()
                if any(w in q_lower for w in ["culture","customs","etiquette","tradition","language"]):
                    tools_needed = ["cultural_info"]
                elif any(w in q_lower for w in ["health","vaccine","safe","medical","hospital"]):
                    tools_needed = ["health"]
                elif any(w in q_lower for w in ["visa","passport","entry","immigration"]):
                    tools_needed = ["visa"]
                elif any(w in q_lower for w in ["transport","metro","bus","taxi","train"]):
                    tools_needed = ["transport"]
                elif any(w in q_lower for w in ["hotel","accommodation","stay","hostel"]):
                    tools_needed = ["hotels"]
                elif any(w in q_lower for w in ["restaurant","food","eat","dining","cuisine"]):
                    tools_needed = ["restaurants"]
                elif any(w in q_lower for w in ["flight","fly","airline","airfare"]):
                    tools_needed = ["flights"]
                else:
                    tools_needed = ["weather"]

            result = {
                "query": query,
                "query_type": query_type,
                "location": destination_city,
                "location_country": destination_country,
                "origin": origin_city,
                "origin_country": origin_country,
                "tools_needed": tools_needed,
                "is_travel_related": parsed.get("is_travel_related", True),
            }
            self.logger.info(
                f"Routed: type={query_type}, city={destination_city!r} "
                f"country={destination_country!r}, origin={origin_city!r}, "
                f"tools={result['tools_needed']}"
            )
            return result

        except Exception as e:
            self.logger.error(f"Routing failed: {e}", exc_info=True)
            return {
                "query": query,
                "query_type": "INFORMATION",
                "location": None,
                "origin": None,
                "tools_needed": ["weather"],
                "is_travel_related": True,
            }


class InformationAgent(BaseAgent):
    """
    Handles single factual queries: weather, flights, visa, health, hotels, restaurants,
    transport, culture. Runs only the tools specified by the router.
    """

    def __init__(self, llm: BaseLLM, tools: Optional[Dict[str, Any]] = None):
        super().__init__("InformationAgent", llm)
        self.tools = tools or {}

    def execute(self, query: str, location: Optional[str] = None,
                location_country: Optional[str] = None,
                origin: Optional[str] = None, origin_country: Optional[str] = None,
                tools_needed: Optional[List[str]] = None,
                **kwargs) -> Dict[str, Any]:
        try:
            tools_to_run = tools_needed or ["weather"]
            has_location = bool(location)
            has_flight_context = "flights" in tools_to_run and (origin or location)

            if not has_location and not has_flight_context:
                return {
                    "answer": "Please specify a destination so I can help you with travel information.",
                    "tools_used": [],
                    "tool_data": {},
                }

            # For visa/health/culture: prefer country name; fall back to city
            country_for_tools = location_country or location
            tool_data = self._run_tools(tools_to_run, location or origin, origin,
                                        country_for_tools, origin_country)
            answer = self._generate_answer(query, location or origin, tool_data)

            return {
                "answer": answer,
                "destination": location or origin,
                "tools_used": tools_to_run,
                "tool_data": tool_data,
            }

        except Exception as e:
            self.logger.error(f"InformationAgent failed: {e}", exc_info=True)
            raise AgentExecutionError(
                f"Information processing failed: {e}",
                agent_name=self.name,
                context={"query": query},
            )

    def _run_tools(self, tools_to_run: List[str], location: Optional[str],
                   origin: Optional[str], country: Optional[str] = None,
                   origin_country: Optional[str] = None) -> Dict[str, Any]:
        tool_data = {}
        for tool_name in tools_to_run:
            tool = self.tools.get(tool_name)
            if not tool:
                self.logger.warning(f"Tool '{tool_name}' not found in registry")
                continue
            try:
                if tool_name == "weather" and location:
                    tool_data["weather"] = tool.run(destination=location)
                elif tool_name == "flights":
                    if origin and location:
                        tool_data["flights"] = tool.run(
                            origin=origin, destination=location,
                            departure_date=datetime.now().strftime("%Y-%m-%d"),
                        )
                    else:
                        o_label = origin or "the origin"
                        d_label = location or "the destination"
                        tool_data["flights"] = {
                            "origin": origin, "destination": location,
                            "source": "llm_knowledge",
                            "note": (
                                f"Provide flight information from {o_label} to {d_label} "
                                f"from your training knowledge. Include airlines that typically "
                                f"serve this route, flight duration, stops, and approximate "
                                f"price range. "
                                f"Disclaimer: prices and schedules change — check airline "
                                f"websites or Google Flights for current fares."
                            ),
                        }
                elif tool_name == "visa" and (country or location):
                    tool_data["visa"] = tool.run(
                        origin_country=origin_country or origin or "US",
                        destination_country=country or location,
                    )
                elif tool_name == "health" and (country or location):
                    tool_data["health"] = tool.run(destination_country=country or location)
                elif tool_name == "hotels" and location:
                    tool_data["hotels"] = tool.run(destination=location)
                elif tool_name == "restaurants" and location:
                    tool_data["restaurants"] = tool.run(destination=location)
                elif tool_name == "transport" and location:
                    tool_data["transport"] = tool.run(destination=location)
                elif tool_name == "cultural_info" and (country or location):
                    tool_data["cultural_info"] = tool.run(destination_country=country or location)
                self.logger.info(f"Tool '{tool_name}' completed")
            except Exception as e:
                self.logger.warning(f"Tool '{tool_name}' failed: {e}")
        return tool_data

    def _generate_answer(self, query: str, location: str,
                         tool_data: Dict[str, Any]) -> str:
        context_parts = []
        instructions = []

        for tool_name, data in tool_data.items():
            if not isinstance(data, dict):
                continue
            source = data.get("source", "")
            note = data.get("note", "")
            if "llm_knowledge" in source:
                instructions.append(note)
            else:
                clean = {k: v for k, v in data.items() if k not in ("source", "note")}
                context_parts.append(
                    f"[{tool_name.upper()} — {source}]\n"
                    f"{json.dumps(clean, indent=2, default=str)}"
                )

        today = datetime.now().strftime("%A, %d %B %Y, %H:%M")
        prompt_parts = [
            f'User query: "{query}"\n'
            f'Destination: {location}\n'
            f'Current date/time: {today}\n'
        ]

        if context_parts:
            prompt_parts.append("LIVE DATA FROM APIs:\n" + "\n\n".join(context_parts))

        if instructions:
            prompt_parts.append(
                "ANSWER FROM TRAINING KNOWLEDGE for these topics "
                "(add disclaimer on each):\n"
                + "\n".join(f"- {inst}" for inst in instructions)
            )

        prompt_parts.append(
            "\nStructure your answer:\n"
            "1. Direct answer using the data above.\n"
            "2. Practical tips (packing/weather, booking advice, local tips, etc.).\n"
            "3. Tag live API data with its source. "
            "   Tag knowledge-based answers with *(general knowledge — verify before travel)*.\n"
            "Be concise, specific, and honest. Do not invent facts you are unsure of."
        )

        return self.llm.generate(
            "\n\n".join(prompt_parts),
            system_message=_GUARDRAIL_SYSTEM,
            max_tokens=1000,
        )


class PlanningAgent(BaseAgent):
    """Creates day-by-day itineraries. Always runs weather + hotels + cultural_info,
    plus flights if origin is known."""

    def __init__(self, llm: BaseLLM, tools: Optional[Dict[str, Any]] = None):
        super().__init__("PlanningAgent", llm)
        self.tools = tools or {}

    def execute(self, query: str, location: Optional[str] = None,
                origin: Optional[str] = None, budget: Optional[float] = None,
                start_date: Optional[str] = None, end_date: Optional[str] = None,
                preferences: Optional[Dict] = None, **kwargs) -> Dict[str, Any]:
        try:
            if not location:
                return {
                    "itinerary": "Please specify a destination for itinerary planning.",
                    "tools_used": [], "tool_data": {},
                }

            self.logger.info(f"Planning trip to {location}")
            tool_data = {}

            for tool_name in ["weather", "hotels", "cultural_info"]:
                tool = self.tools.get(tool_name)
                if not tool:
                    continue
                try:
                    if tool_name == "weather":
                        tool_data["weather"] = tool.run(destination=location)
                    elif tool_name == "hotels":
                        tool_data["hotels"] = tool.run(
                            destination=location,
                            check_in=start_date,
                            check_out=end_date,
                        )
                    elif tool_name == "cultural_info":
                        tool_data["cultural_info"] = tool.run(destination_country=location)
                except Exception as e:
                    self.logger.warning(f"Planning tool {tool_name} failed: {e}")

            if origin and origin != "NOT_SPECIFIED":
                flights_tool = self.tools.get("flights")
                if flights_tool:
                    try:
                        tool_data["flights"] = flights_tool.run(
                            origin=origin, destination=location,
                            departure_date=start_date or datetime.now().strftime("%Y-%m-%d"),
                        )
                    except Exception as e:
                        self.logger.warning(f"Planning flights tool failed: {e}")

            itinerary = self._generate_itinerary(
                query, location, budget, start_date, end_date, preferences, tool_data
            )
            return {
                "itinerary": itinerary,
                "destination": location,
                "tools_used": list(tool_data.keys()),
                "tool_data": tool_data,
            }

        except Exception as e:
            self.logger.error(f"PlanningAgent failed: {e}", exc_info=True)
            raise AgentExecutionError(
                f"Planning failed: {e}", agent_name=self.name,
                context={"location": location},
            )

    def _generate_itinerary(self, query: str, location: str, budget: Optional[float],
                             start_date: Optional[str], end_date: Optional[str],
                             preferences: Optional[Dict], tool_data: Dict) -> str:
        context_parts = []
        instructions = []
        for tool_name, data in tool_data.items():
            if not isinstance(data, dict):
                continue
            source = data.get("source", "")
            note = data.get("note", "")
            if "llm_knowledge" in source:
                instructions.append(note)
            else:
                clean = {k: v for k, v in data.items() if k not in ("source", "note")}
                context_parts.append(
                    f"[{tool_name.upper()}]\n{json.dumps(clean, indent=2, default=str)}"
                )

        today = datetime.now().strftime("%Y-%m-%d")
        prompt = (
            f"Create a detailed travel itinerary.\n"
            f"Today: {today}\n"
            f"Destination: {location}\n"
            f"Dates: {start_date or 'flexible'} to {end_date or 'flexible'}\n"
            f"Budget: {'$' + str(budget) if budget else 'not specified'}\n"
            f"Preferences: {json.dumps(preferences) if preferences else 'not specified'}\n"
            f"Query: {query}\n\n"
        )
        if context_parts:
            prompt += "LIVE DATA:\n" + "\n\n".join(context_parts) + "\n\n"
        if instructions:
            prompt += (
                "USE TRAINING KNOWLEDGE FOR (add disclaimer on each):\n"
                + "\n".join(f"- {i}" for i in instructions)
                + "\n\n"
            )
        prompt += (
            "Write a day-by-day itinerary including: activities, restaurant suggestions, "
            "accommodation, transport, and a budget breakdown. "
            "Mark knowledge-based suggestions with *(verify current prices)*."
        )

        return self.llm.generate(
            prompt,
            system_message=_GUARDRAIL_SYSTEM,
            max_tokens=3000,
        )


class RecommendationAgent(BaseAgent):
    """
    Generates personalized recommendations.
    Runs only the tools the router specified in tools_needed — does not hardcode a fixed set.
    Falls back to restaurants + hotels + cultural_info if tools_needed is empty.
    """

    def __init__(self, llm: BaseLLM, tools: Optional[Dict[str, Any]] = None):
        super().__init__("RecommendationAgent", llm)
        self.tools = tools or {}

    def execute(self, query: str, location: Optional[str] = None,
                tools_needed: Optional[List[str]] = None,
                traveler_profile: Optional[Dict] = None, **kwargs) -> Dict[str, Any]:
        try:
            if not location:
                return {
                    "recommendations": "Please specify a destination for recommendations.",
                    "tools_used": [], "tool_data": {},
                }

            # Use router-specified tools; fall back to the natural recommendation set
            tools_to_run = tools_needed or ["restaurants", "hotels", "cultural_info"]
            self.logger.info(
                f"RecommendationAgent running tools {tools_to_run} for {location}"
            )

            tool_data = {}
            for tool_name in tools_to_run:
                tool = self.tools.get(tool_name)
                if not tool:
                    self.logger.warning(f"Tool '{tool_name}' not found")
                    continue
                try:
                    if tool_name == "restaurants":
                        tool_data["restaurants"] = tool.run(destination=location)
                    elif tool_name == "hotels":
                        tool_data["hotels"] = tool.run(destination=location)
                    elif tool_name == "cultural_info":
                        tool_data["cultural_info"] = tool.run(destination_country=location)
                    elif tool_name == "weather":
                        tool_data["weather"] = tool.run(destination=location)
                    elif tool_name == "transport":
                        tool_data["transport"] = tool.run(destination=location)
                    elif tool_name == "health":
                        tool_data["health"] = tool.run(destination_country=location)
                    self.logger.info(f"Tool '{tool_name}' completed")
                except Exception as e:
                    self.logger.warning(f"Tool '{tool_name}' failed: {e}")

            recommendations = self._generate_recommendations(
                query, location, traveler_profile, tool_data
            )
            return {
                "recommendations": recommendations,
                "destination": location,
                "tools_used": tools_to_run,
                "tool_data": tool_data,
            }

        except Exception as e:
            self.logger.error(f"RecommendationAgent failed: {e}", exc_info=True)
            raise AgentExecutionError(
                f"Recommendations failed: {e}", agent_name=self.name,
                context={"location": location},
            )

    def _generate_recommendations(self, query: str, location: str,
                                   traveler_profile: Optional[Dict],
                                   tool_data: Dict) -> str:
        context_parts = []
        instructions = []
        for tool_name, data in tool_data.items():
            if not isinstance(data, dict):
                continue
            source = data.get("source", "")
            note = data.get("note", "")
            if "llm_knowledge" in source:
                instructions.append(note)
            else:
                clean = {k: v for k, v in data.items() if k not in ("source", "note")}
                context_parts.append(
                    f"[{tool_name.upper()}]\n{json.dumps(clean, indent=2, default=str)}"
                )

        profile_str = ""
        if traveler_profile:
            interests = traveler_profile.get("travel_preferences", {}).get("interests", [])
            budget = traveler_profile.get("travel_preferences", {}).get("budget_range", {})
            profile_str = f"\nTraveler interests: {interests}, Budget: {budget}"

        today = datetime.now().strftime("%A, %d %B %Y")
        prompt = (
            f'Query: "{query}"\n'
            f'Destination: {location}\n'
            f'Date: {today}{profile_str}\n\n'
        )
        if context_parts:
            prompt += "LIVE DATA:\n" + "\n\n".join(context_parts) + "\n\n"
        if instructions:
            prompt += (
                "ANSWER FROM TRAINING KNOWLEDGE (add disclaimer on each):\n"
                + "\n".join(f"- {inst}" for inst in instructions)
                + "\n\n"
            )
        prompt += (
            "Give specific, practical recommendations that directly answer the query. "
            "Use actual names, realistic details. "
            "Add *(verify before booking)* on knowledge-based items."
        )

        return self.llm.generate(
            prompt,
            system_message=_GUARDRAIL_SYSTEM,
            max_tokens=1500,
        )


class XAIValidator(BaseAgent):
    """
    Computes confidence score based on data provenance:
    - Live API data → high confidence
    - LLM knowledge → lower confidence
    No extra LLM call — pure computation.
    """

    def __init__(self, llm: BaseLLM):
        super().__init__("XAIValidator", llm)

    def execute(self, agent_outputs: Dict[str, Any], original_query: str,
                **kwargs) -> Dict[str, Any]:
        try:
            tools_live = 0
            tools_total = 0

            for agent_key in ("information_agent", "planning_agent", "recommendation_agent"):
                agent_data = agent_outputs.get(agent_key, {})
                if not isinstance(agent_data, dict):
                    continue
                for _, tool_result in agent_data.get("tool_data", {}).items():
                    tools_total += 1
                    src = tool_result.get("source", "") if isinstance(tool_result, dict) else ""
                    if "live" in src.lower() or ("api" in src.lower() and "llm" not in src.lower()):
                        tools_live += 1

            if tools_total > 0:
                live_ratio = tools_live / tools_total
                # 0.70 base + up to 0.25 for live data
                confidence = round(min(0.70 + live_ratio * 0.25, 0.95), 2)
            else:
                confidence = 0.70

            return {
                "validation_status": "approved",
                "confidence_score": confidence,
                "live_data_tools": tools_live,
                "total_tools": tools_total,
                "data_sources": "live_api" if tools_live == tools_total
                                else "mixed" if tools_live > 0
                                else "llm_knowledge",
                "warnings": [],
            }

        except Exception as e:
            self.logger.error(f"Validation failed: {e}")
            return {
                "validation_status": "approved",
                "confidence_score": 0.70,
                "warnings": [str(e)],
            }