"""
Common module: Shared utilities, exceptions, configuration, and logging.
"""

from .exceptions import (
    TravelAIException,
    ConfigurationError,
    AgentExecutionError,
    ToolExecutionError,
    LLMError,
    APIError,
    DatabaseError,
    RAGError,
    ValidationError,
    TimeoutError,
)
from .logger import (
    configure_logging,
    get_logger,
    get_agent_logger,
    get_tool_logger,
    get_api_logger,
    log_api_call,
    log_suppressed_exception,
    LoggerConfig,
)
from .settings import get_settings, reload_settings, Settings

__all__ = [
    # Exceptions
    "TravelAIException",
    "ConfigurationError",
    "AgentExecutionError",
    "ToolExecutionError",
    "LLMError",
    "APIError",
    "DatabaseError",
    "RAGError",
    "ValidationError",
    "TimeoutError",
    # Logging
    "configure_logging",
    "get_logger",
    "get_agent_logger",
    "get_tool_logger",
    "get_api_logger",
    "log_api_call",
    "log_suppressed_exception",
    "LoggerConfig",
    # Configuration
    "get_settings",
    "reload_settings",
    "Settings",
]
