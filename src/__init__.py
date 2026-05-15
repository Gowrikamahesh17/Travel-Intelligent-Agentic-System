"""
Travel Intelligent Agentic System
===================================

A sophisticated multi-agent system for intelligent travel planning.

Features:
- Five-agent architecture (Router, Information, Planning, Recommendation, XAI)
- RAG integration with ChromaDB for semantic search
- Support for multiple LLM providers (Gemini, OpenAI, Ollama)
- Comprehensive tool ecosystem (weather, flights, hotels, visa, health, etc.)
- Streamlit-based web UI
- SQLite database with SQLAlchemy ORM
- Advanced logging and error handling

Usage:
    from src.main import initialize_system
    
    system = initialize_system()
    result = system.process_query("Plan a 7-day trip to Japan with $3000 budget")
"""

__version__ = "1.0.0"
__author__ = "Travel AI Team"

from src.main import initialize_system, get_system

__all__ = ["initialize_system", "get_system"]
