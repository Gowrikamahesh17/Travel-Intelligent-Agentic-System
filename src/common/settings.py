"""
Centralized configuration management using Pydantic.
Environment-based configuration with support for multiple providers and feature flags.
"""

import os
from pathlib import Path
from pydantic_settings import BaseSettings
from typing import Optional, Literal


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    Supports easy switching between LLM providers and feature flags.
    """

    # ==================== HuggingFace Configuration ====================
    HF_TOKEN: Optional[str] = None
    
    # ==================== LLM Configuration ====================
    PRIMARY_LLM_PROVIDER: Literal["gemini", "openai", "ollama"] = "gemini"
    GEMINI_API_KEY: Optional[str] = None
    GEMINI_MODEL: str = "gemini-2.5-flash"
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: str = "gpt-3.5-turbo"
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama2"
    LLM_TEMPERATURE: float = 0.7

    # ==================== Feature Flags ====================
    ENABLE_RAG: bool = True
    ENABLE_XAI_VALIDATOR: bool = True
    ENABLE_RECOMMENDATIONS: bool = True
    USE_MOCK_EMBEDDINGS: bool = False

    # ==================== RAG Configuration ====================
    EMBEDDINGS_PROVIDER: Literal["mock", "openai", "huggingface"] = "huggingface"
    CHROMA_COLLECTION_PREFIX: str = "travel_ai"
    CHROMA_DB_PATH: Optional[str] = None  # Will default to project root if not set

    # ==================== Database Configuration ====================
    DATABASE_URL: str = None  # Will default to project root if not set
    LOG_DATABASE_QUERIES: bool = False

    # ==================== API Configuration ====================
    API_TIMEOUT: int = 30
    API_MAX_RETRIES: int = 3
    RETRY_BACKOFF_FACTOR: float = 2.0

    # ==================== Logging Configuration ====================
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: Literal["json", "text"] = "json"
    LOG_DIR: str = "logs"  # Will be converted to absolute path

    # ==================== Cache Configuration ====================
    CACHE_TTL_SECONDS: int = 86400  # 24 hours

    # ==================== External APIs ====================
    WEATHER_API_KEY: Optional[str] = None
    WEATHER_API_BASE_URL: str = "https://api.openweathermap.org/data/2.5"
    BOOKING_API_KEY: Optional[str] = None
    HOTEL_API_KEY: Optional[str] = None
    VISA_API_KEY: Optional[str] = None
    HEALTH_API_KEY: Optional[str] = None

    # ==================== Development ====================
    DEBUG: bool = False
    ENVIRONMENT: Literal["development", "staging", "production"] = "production"

    class Config:
        """Pydantic configuration."""
        # Use absolute path to .env file from project root
        env_file = str(Path(__file__).parent.parent.parent / ".env")
        env_file_encoding = "utf-8"
        case_sensitive = True

    def __init__(self, **data):
        """Initialize settings and normalize paths."""
        super().__init__(**data)
        
        # Normalize DATABASE_URL
        if not self.DATABASE_URL or self.DATABASE_URL == "sqlite:///./travel_ai.db":
            db_path = Path(__file__).parent.parent.parent / "travel_ai.db"
            self.DATABASE_URL = f"sqlite:///{db_path}"
        
        # Normalize LOG_DIR
        if self.LOG_DIR and not Path(self.LOG_DIR).is_absolute():
            log_path = Path(__file__).parent.parent.parent / self.LOG_DIR
            self.LOG_DIR = str(log_path)
        
        # Normalize CHROMA_DB_PATH
        if not self.CHROMA_DB_PATH or self.CHROMA_DB_PATH == "./chroma_db":
            chroma_path = Path(__file__).parent.parent.parent / "chroma_db"
            self.CHROMA_DB_PATH = str(chroma_path)

    def validate_settings(self) -> None:
        """
        Validate critical settings and raise errors if misconfigured.

        Raises:
            ValueError: If critical settings are missing.
        """
        if self.PRIMARY_LLM_PROVIDER == "gemini" and not self.GEMINI_API_KEY:
            raise ValueError(
                "GEMINI_API_KEY must be set when PRIMARY_LLM_PROVIDER=gemini"
            )
        if self.PRIMARY_LLM_PROVIDER == "openai" and not self.OPENAI_API_KEY:
            raise ValueError(
                "OPENAI_API_KEY must be set when PRIMARY_LLM_PROVIDER=openai"
            )

        if self.EMBEDDINGS_PROVIDER == "openai" and not self.OPENAI_API_KEY:
            raise ValueError(
                "OPENAI_API_KEY must be set when EMBEDDINGS_PROVIDER=openai"
            )


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """
    Get global settings instance.
    Lazy loads and validates settings on first access.

    Returns:
        Settings instance
    """
    global _settings
    if _settings is None:
        _settings = Settings()
        _settings.validate_settings()
    return _settings


def reload_settings() -> Settings:
    """Force reload settings (useful for testing)."""
    global _settings
    _settings = Settings()
    _settings.validate_settings()
    return _settings
