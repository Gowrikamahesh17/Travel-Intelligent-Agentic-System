"""
Agents module: Five-agent architecture with LangGraph orchestration.
QueryRouter, InformationAgent, PlanningAgent, RecommendationAgent, XAIValidator.
"""

from .agents import (
    BaseAgent,
    QueryRouter,
    InformationAgent,
    PlanningAgent,
    RecommendationAgent,
    XAIValidator,
)
from .orchestrator import AgentOrchestrator          # legacy, kept for compatibility
from .graph import LangGraphOrchestrator, build_travel_graph  # primary

__all__ = [
    "BaseAgent",
    "QueryRouter",
    "InformationAgent",
    "PlanningAgent",
    "RecommendationAgent",
    "XAIValidator",
    "AgentOrchestrator",
    "LangGraphOrchestrator",
    "build_travel_graph",
]
