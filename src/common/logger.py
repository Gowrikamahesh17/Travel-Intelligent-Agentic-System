"""
Centralized logging configuration for Travel AI System.
Supports multiple output formats and agent-specific tracking.
"""

import logging
import logging.handlers
import sys
import json
from pathlib import Path
from typing import Optional
from pythonjsonlogger import jsonlogger


class LoggerConfig:
    """Centralized logging configuration."""

    def __init__(
        self,
        log_level: str = "INFO",
        log_dir: str = None,
        log_format: str = "json",
        enable_console: bool = True,
        enable_file: bool = True,
    ):
        self.log_level = log_level
        # Use absolute path: project root / logs
        if log_dir is None:
            log_dir = str(Path(__file__).parent.parent.parent / "logs")
        else:
            # Convert relative paths to absolute
            log_path = Path(log_dir)
            if not log_path.is_absolute():
                log_dir = str(Path(__file__).parent.parent.parent / log_dir)
        
        self.log_dir = log_dir
        self.log_format = log_format
        self.enable_console = enable_console
        self.enable_file = enable_file

        # Create logs directory if it doesn't exist
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)

    def get_logger(self, name: str, agent_name: Optional[str] = None) -> logging.Logger:
        """
        Get configured logger instance.

        Args:
            name: Logger name (typically __name__)
            agent_name: Optional agent name for context

        Returns:
            Configured logger instance
        """
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, self.log_level))

        # Clear existing handlers to avoid duplicates
        logger.handlers.clear()

        # Console handler (stdout streaming)
        if self.enable_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(getattr(logging, self.log_level))

            if self.log_format == "json":
                formatter = jsonlogger.JsonFormatter(
                    "%(asctime)s %(levelname)s %(name)s %(message)s"
                )
            else:
                formatter = logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                )
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

        # File handler with rotation for all logs
        if self.enable_file:
            log_file = Path(self.log_dir) / f"{name.replace('.', '_')}.log"
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=10 * 1024 * 1024,  # 10MB
                backupCount=10,
            )
            file_handler.setLevel(getattr(logging, self.log_level))

            if self.log_format == "json":
                formatter = jsonlogger.JsonFormatter(
                    "%(asctime)s %(levelname)s %(name)s %(message)s"
                )
            else:
                formatter = logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

            # Separate ERROR+ log file for critical issues
            error_log_file = Path(self.log_dir) / "error.log"
            error_handler = logging.handlers.RotatingFileHandler(
                error_log_file,
                maxBytes=10 * 1024 * 1024,  # 10MB
                backupCount=10,
            )
            error_handler.setLevel(logging.ERROR)

            if self.log_format == "json":
                formatter = jsonlogger.JsonFormatter(
                    "%(asctime)s %(levelname)s %(name)s %(message)s %(exc_info)s"
                )
            else:
                formatter = logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s - %(exc_info)s"
                )
            error_handler.setFormatter(formatter)
            logger.addHandler(error_handler)

        return logger

    def get_agent_logger(self, agent_name: str) -> logging.Logger:
        """Get logger for specific agent."""
        logger = self.get_logger(f"agents.{agent_name}", agent_name=agent_name)
        return logger

    def get_tool_logger(self, tool_name: str) -> logging.Logger:
        """Get logger for specific tool."""
        logger = self.get_logger(f"tools.{tool_name}")
        return logger

    def get_api_logger(self) -> logging.Logger:
        """Get logger for API calls with full payload tracking."""
        api_logger = self.get_logger("api_calls")
        return api_logger


# Global logger configuration instance
_logger_config: Optional[LoggerConfig] = None


def configure_logging(
    log_level: str = "INFO",
    log_dir: str = "./logs",
    log_format: str = "json",
) -> LoggerConfig:
    """
    Initialize global logging configuration.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files
        log_format: Log format ("json" or "text")

    Returns:
        LoggerConfig instance
    """
    global _logger_config
    _logger_config = LoggerConfig(
        log_level=log_level,
        log_dir=log_dir,
        log_format=log_format,
    )
    return _logger_config


def get_logger(name: str) -> logging.Logger:
    """Get logger instance (uses global config)."""
    if _logger_config is None:
        configure_logging()
    return _logger_config.get_logger(name)


def get_agent_logger(agent_name: str) -> logging.Logger:
    """Get logger for agent."""
    if _logger_config is None:
        configure_logging()
    return _logger_config.get_agent_logger(agent_name)


def get_tool_logger(tool_name: str) -> logging.Logger:
    """Get logger for tool."""
    if _logger_config is None:
        configure_logging()
    return _logger_config.get_tool_logger(tool_name)


def get_api_logger() -> logging.Logger:
    """Get logger for API calls with full payload tracking."""
    if _logger_config is None:
        configure_logging()
    return _logger_config.get_api_logger()


def log_api_call(
    provider: str,
    endpoint: str,
    method: str,
    payload: dict,
    response: Optional[dict] = None,
    error: Optional[str] = None,
) -> None:
    """
    Log API call with full details.

    Args:
        provider: API provider name (e.g., "gemini", "openai")
        endpoint: API endpoint
        method: HTTP method
        payload: Request payload
        response: Response data (if successful)
        error: Error message (if failed)
    """
    api_logger = get_api_logger()
    
    log_entry = {
        "provider": provider,
        "endpoint": endpoint,
        "method": method,
        "payload": payload,
        "response": response,
        "error": error,
    }
    
    if error:
        api_logger.error(f"API call failed: {json.dumps(log_entry, default=str)}")
    else:
        api_logger.info(f"API call successful: {json.dumps(log_entry, default=str)}")


def log_suppressed_exception(
    context: str,
    exception: Exception,
    logger_obj: Optional[logging.Logger] = None,
) -> None:
    """
    Log exceptions that would otherwise be suppressed.

    Args:
        context: Context where exception occurred
        exception: The exception that was suppressed
        logger_obj: Logger instance to use (defaults to main logger)
    """
    if logger_obj is None:
        logger_obj = get_logger(__name__)
    
    logger_obj.warning(
        f"Suppressed exception in {context}: {type(exception).__name__}: {str(exception)}",
        exc_info=exception,
    )
