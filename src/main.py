"""
Main entry point and API orchestrator for Travel Intelligent Agentic System.
Initializes all components and provides unified interface.
"""

import logging
import os
import uuid
from datetime import datetime
from typing import Optional, Dict, Any

# Suppress unnecessary transformer warnings before importing
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("transformers.image_processing_utils").setLevel(logging.ERROR)
logging.getLogger("transformers.utils.generic").setLevel(logging.ERROR)

from src.common import (
    get_logger,
    configure_logging,
    get_settings,
    ConfigurationError,
)
from src.database import get_db_manager, get_session
from src.database.models import UserProfile, QueryHistory
from src.llm import LLMFactory
from src.agents import LangGraphOrchestrator as AgentOrchestrator
from src.tools import get_all_tools
from src.rag import RAGManager, create_embeddings_provider

logger = get_logger(__name__)


def print_startup_banner():
    """Print startup banner to console."""
    banner = """
============================================================
     Travel Intelligent Agentic System
     Multi-Agent AI Travel Planning Assistant
============================================================
"""
    print(banner)
    logger.info("="*60)
    logger.info("Travel Intelligent Agentic System Starting")
    logger.info("="*60)


class TravelAISystem:
    """
    Main Travel AI System class.
    Orchestrates all components: database, LLM, agents, tools, RAG.
    """

    def __init__(self):
        """Initialize the Travel AI System."""
        print_startup_banner()
        try:
            # Load settings
            self.settings = get_settings()
            logger.info(f"Settings loaded. Provider: {self.settings.PRIMARY_LLM_PROVIDER}")
            
            # Set HuggingFace token if configured
            if self.settings.HF_TOKEN:
                os.environ["HF_TOKEN"] = self.settings.HF_TOKEN
                logger.info("HuggingFace token configured from settings")

            # Configure logging
            configure_logging(
                log_level=self.settings.LOG_LEVEL,
                log_dir=self.settings.LOG_DIR,
                log_format=self.settings.LOG_FORMAT,
            )
            logger.info("Logging configured")

            # Initialize database
            get_db_manager().initialize(self.settings.DATABASE_URL)
            logger.info("Database initialized")

            # Initialize LLM
            self.llm = LLMFactory.create_from_settings(self.settings)
            logger.info(f"LLM initialized: {self.llm}")

            # Initialize tools
            self.tools = get_all_tools()
            logger.info(f"Tools initialized: {list(self.tools.keys())}")

            # Initialize RAG
            if self.settings.ENABLE_RAG:
                try:
                    embeddings_kwargs = {}
                    
                    # Only add provider-specific kwargs
                    if self.settings.EMBEDDINGS_PROVIDER == "openai":
                        if not self.settings.OPENAI_API_KEY:
                            logger.warning("OPENAI_API_KEY not set, falling back to mock embeddings")
                            embeddings = create_embeddings_provider("mock")
                        else:
                            embeddings_kwargs["api_key"] = self.settings.OPENAI_API_KEY
                            embeddings = create_embeddings_provider(
                                self.settings.EMBEDDINGS_PROVIDER,
                                **embeddings_kwargs
                            )
                    elif self.settings.EMBEDDINGS_PROVIDER == "huggingface":
                        # HuggingFace embeddings don't need API key for local models
                        embeddings = create_embeddings_provider("huggingface")
                    else:
                        embeddings = create_embeddings_provider(self.settings.EMBEDDINGS_PROVIDER)
                    
                    self.rag_manager = RAGManager(
                        embeddings,
                        collection_prefix=self.settings.CHROMA_COLLECTION_PREFIX,
                        db_path=self.settings.CHROMA_DB_PATH,
                    )
                    logger.info("RAG manager initialized")
                except Exception as e:
                    logger.warning(f"RAG initialization failed: {e}. Continuing without RAG.")
                    self.rag_manager = None
            else:
                self.rag_manager = None
                logger.info("RAG disabled")

            # Initialize agent orchestrator
            self.orchestrator = AgentOrchestrator(
                llm=self.llm,
                tools=self.tools,
                enable_rag=self.settings.ENABLE_RAG,
            )
            logger.info("Agent orchestrator initialized")

            logger.info("✅ Travel AI System initialized successfully!")

        except Exception as e:
            logger.error(f"Failed to initialize system: {str(e)}")
            raise

    def process_query(
        self,
        query: str,
        user_id: str = "default_user",
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Process user query through the agent orchestrator.

        Args:
            query: User query
            user_id: User identifier
            context: Additional context

        Returns:
            Final response with agent trace
        """
        try:
            logger.info(f"Processing query from user {user_id}: {query}")

            # Get user profile
            with get_db_manager().session_context() as session:
                user = session.query(UserProfile).filter(UserProfile.id == user_id).first()
                user_context = None
                if user:
                    user_context = {
                        "name": user.name,
                        "nationality": user.nationality,
                        "preferences": user.travel_preferences,
                        "constraints": {
                            "mobility": user.mobility_constraints,
                            "health": user.health_conditions,
                        },
                    }

            # Merge context into user_context so graph can access budget, dates etc.
            merged_context = dict(user_context or {})
            merged_context.update(context or {})

            # Execute through orchestrator
            result = self.orchestrator.execute(
                query=query,
                user_context=merged_context,
            )

            # Store query in database
            self._store_query(user_id, query, result)

            return result

        except Exception as e:
            logger.error(f"Query processing failed: {str(e)}")
            raise

    def _store_query(
        self, user_id: str, query: str, result: Dict[str, Any]
    ) -> None:
        """Store query history in database."""
        try:
            with get_db_manager().session_context() as session:
                query_record = QueryHistory(
                    id=str(uuid.uuid4()),
                    user_id=user_id,
                    query_text=query,
                    query_type=result["agent_trace"].get("query_router", {}).get("query_type", "unknown"),
                    agents_involved=list(result["agent_trace"].keys()),
                    execution_trace=result["agent_trace"],
                    result={"response": result.get("response", "")},
                    confidence_score=result["validation"].get("confidence_score", 0),
                    execution_time_ms=result.get("execution_time_ms", 0),
                    success=True,
                )
                session.add(query_record)
            logger.info(f"Query stored for user {user_id}")
        except Exception as e:
            logger.warning(f"Failed to store query: {str(e)}")

    def create_user_profile(
        self,
        user_id: str,
        name: str,
        nationality: str,
        email: str,
        preferences: Optional[Dict[str, Any]] = None,
    ) -> UserProfile:
        """
        Create new user profile.

        Args:
            user_id: User identifier
            name: User name
            nationality: User nationality
            email: User email
            preferences: Travel preferences

        Returns:
            Created user profile
        """
        try:
            with get_db_manager().session_context() as session:
                user = UserProfile(
                    id=user_id,
                    name=name,
                    nationality=nationality,
                    email=email,
                    travel_preferences=preferences or {},
                )
                session.add(user)
            logger.info(f"Created user profile: {user_id}")
            return user
        except Exception as e:
            logger.error(f"Failed to create user profile: {str(e)}")
            raise

    def get_tool(self, tool_name: str):
        """Get tool by name."""
        return self.tools.get(tool_name)

    def get_system_status(self) -> Dict[str, Any]:
        """Get system status."""
        return {
            "status": "ready",
            "llm_provider": self.settings.PRIMARY_LLM_PROVIDER,
            "rag_enabled": self.settings.ENABLE_RAG,
            "xai_enabled": self.settings.ENABLE_XAI_VALIDATOR,
            "tools_available": len(self.tools),
            "agents": self.orchestrator.get_agent_status(),
        }


# Global system instance
_system: Optional[TravelAISystem] = None


def initialize_system() -> TravelAISystem:
    """Initialize and return global system instance."""
    global _system
    if _system is None:
        _system = TravelAISystem()
    return _system


def get_system() -> TravelAISystem:
    """Get global system instance."""
    global _system
    if _system is None:
        _system = TravelAISystem()
    return _system


# UI Convenience function
def process_query(
    query: str,
    user_id: str = "default_user",
    destination: Optional[str] = None,
    budget: Optional[int] = None,
    user_profile: Optional[Dict[str, Any]] = None,
    context: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Convenience function for UI to process travel queries."""
    system = get_system()
    # Merge all UI-level params into context so orchestrator can use them
    merged_context = dict(context or {})
    if destination:
        merged_context["destination"] = destination
    if budget:
        merged_context["budget"] = budget
    if user_profile:
        merged_context.update(user_profile)
    return system.process_query(query=query, user_id=user_id, context=merged_context)


if __name__ == "__main__":
    # Example usage
    system = initialize_system()

    # Create a test user
    system.create_user_profile(
        user_id="test_user",
        name="John Doe",
        nationality="US",
        email="john@example.com",
        preferences={
            "budget_range": {"min": 1000, "max": 5000},
            "trip_duration": 7,
        },
    )

    # Process a sample query
    result = system.process_query(
        query="Plan a 7-day trip to Japan with a $3000 budget",
        user_id="test_user",
        context={
            "destination": "Japan",
            "start_date": "2024-05-20",
            "end_date": "2024-05-27",
            "budget": 3000,
        },
    )

    print("\n[SUCCESS] Query processed successfully!")
    print(f"Response: {result['response'][:200]}...")

