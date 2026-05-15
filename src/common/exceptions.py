"""
Custom exception hierarchy for Travel Intelligent Agentic System.
Provides structured error handling with context and retry information.
"""

from typing import Optional, Dict, Any


class TravelAIException(Exception):
    """Base exception for all Travel AI System errors."""

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        retryable: bool = False,
    ):
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.context = context or {}
        self.retryable = retryable
        super().__init__(self.message)

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging/response."""
        return {
            "error_code": self.error_code,
            "message": self.message,
            "context": self.context,
            "retryable": self.retryable,
        }


class ConfigurationError(TravelAIException):
    """Raised when configuration is missing or invalid."""

    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message, "CONFIG_ERROR", context, retryable=False)


class AgentExecutionError(TravelAIException):
    """Raised when an agent fails to execute."""

    def __init__(
        self,
        message: str,
        agent_name: str,
        context: Optional[Dict[str, Any]] = None,
        retryable: bool = True,
    ):
        context = context or {}
        context["agent_name"] = agent_name
        super().__init__(message, "AGENT_ERROR", context, retryable)


class ToolExecutionError(TravelAIException):
    """Raised when a tool execution fails."""

    def __init__(
        self,
        message: str,
        tool_name: str,
        context: Optional[Dict[str, Any]] = None,
        retryable: bool = True,
    ):
        context = context or {}
        context["tool_name"] = tool_name
        super().__init__(message, "TOOL_ERROR", context, retryable)


class LLMError(TravelAIException):
    """Raised when LLM provider fails."""

    def __init__(
        self,
        message: str,
        provider: str,
        context: Optional[Dict[str, Any]] = None,
        retryable: bool = True,
    ):
        context = context or {}
        context["provider"] = provider
        super().__init__(message, "LLM_ERROR", context, retryable)


class APIError(TravelAIException):
    """Raised when external API call fails."""

    def __init__(
        self,
        message: str,
        api_name: str,
        status_code: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None,
        retryable: bool = True,
    ):
        context = context or {}
        context["api_name"] = api_name
        context["status_code"] = status_code
        super().__init__(message, "API_ERROR", context, retryable)


class DatabaseError(TravelAIException):
    """Raised when database operation fails."""

    def __init__(
        self,
        message: str,
        operation: str,
        context: Optional[Dict[str, Any]] = None,
        retryable: bool = True,
    ):
        context = context or {}
        context["operation"] = operation
        super().__init__(message, "DB_ERROR", context, retryable)


class RAGError(TravelAIException):
    """Raised when RAG/vector database operation fails."""

    def __init__(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        retryable: bool = True,
    ):
        super().__init__(message, "RAG_ERROR", context, retryable)


class ValidationError(TravelAIException):
    """Raised when input validation fails."""

    def __init__(self, message: str, field: Optional[str] = None):
        context = {"field": field} if field else {}
        super().__init__(message, "VALIDATION_ERROR", context, retryable=False)


class TimeoutError(TravelAIException):
    """Raised when operation times out."""

    def __init__(
        self, message: str, timeout_seconds: float, context: Optional[Dict[str, Any]] = None
    ):
        context = context or {}
        context["timeout_seconds"] = timeout_seconds
        super().__init__(message, "TIMEOUT_ERROR", context, retryable=True)
