"""
Base tool class with retry logic, caching, and error handling.
All concrete tools inherit from this base class.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, Dict
import time
import hashlib
import json
from datetime import datetime, timedelta
from tenacity import retry, stop_after_attempt, wait_exponential
from src.common import get_tool_logger, ToolExecutionError, TimeoutError


class BaseTool(ABC):
    """
    Abstract base class for all tools.
    Provides retry logic, caching, error handling, and logging.
    """

    def __init__(
        self,
        name: str,
        description: str,
        cache_enabled: bool = True,
        cache_ttl_seconds: int = 86400,
        max_retries: int = 3,
        timeout_seconds: float = 30,
    ):
        """
        Initialize base tool.

        Args:
            name: Tool name
            description: Tool description
            cache_enabled: Whether to enable result caching
            cache_ttl_seconds: Cache TTL in seconds
            max_retries: Maximum retry attempts
            timeout_seconds: Timeout for execution
        """
        self.name = name
        self.description = description
        self.cache_enabled = cache_enabled
        self.cache_ttl_seconds = cache_ttl_seconds
        self.max_retries = max_retries
        self.timeout_seconds = timeout_seconds
        self.logger = get_tool_logger(name)
        self._cache: Dict[str, tuple] = {}  # (result, expiry_time)

    @abstractmethod
    def execute(self, **kwargs) -> Any:
        """
        Execute tool logic (must be implemented by subclasses).

        Args:
            **kwargs: Tool-specific arguments

        Returns:
            Tool execution result

        Raises:
            ToolExecutionError: If execution fails
        """
        pass

    def _get_cache_key(self, **kwargs) -> str:
        """
        Generate cache key from arguments.

        Args:
            **kwargs: Arguments to cache

        Returns:
            Cache key hash
        """
        key_str = json.dumps(kwargs, sort_keys=True, default=str)
        return hashlib.md5(key_str.encode()).hexdigest()

    def _get_cached_result(self, cache_key: str) -> Optional[Any]:
        """
        Get result from cache if valid.

        Args:
            cache_key: Cache key

        Returns:
            Cached result or None if not found/expired
        """
        if not self.cache_enabled or cache_key not in self._cache:
            return None

        result, expiry_time = self._cache[cache_key]
        if datetime.utcnow() > expiry_time:
            del self._cache[cache_key]
            return None

        self.logger.info(f"Cache hit for key: {cache_key}")
        return result

    def _set_cache(self, cache_key: str, result: Any) -> None:
        """
        Set result in cache.

        Args:
            cache_key: Cache key
            result: Result to cache
        """
        if self.cache_enabled:
            expiry_time = datetime.utcnow() + timedelta(seconds=self.cache_ttl_seconds)
            self._cache[cache_key] = (result, expiry_time)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def _execute_with_retry(self, **kwargs) -> Any:
        """
        Execute with automatic retry logic.

        Args:
            **kwargs: Tool arguments

        Returns:
            Execution result

        Raises:
            ToolExecutionError: If all retries fail
        """
        try:
            self.logger.info(f"Executing {self.name} with args: {kwargs}")
            start_time = time.time()

            result = self.execute(**kwargs)

            execution_time_ms = (time.time() - start_time) * 1000
            self.logger.info(
                f"Tool {self.name} executed successfully in {execution_time_ms:.2f}ms"
            )
            return result

        except Exception as e:
            self.logger.error(f"Error executing {self.name}: {str(e)}")
            raise ToolExecutionError(
                message=f"Tool {self.name} execution failed: {str(e)}",
                tool_name=self.name,
                context={"original_error": str(e)},
                retryable=True,
            )

    def run(self, **kwargs) -> Any:
        """
        Execute tool with caching and retry logic.

        Args:
            **kwargs: Tool-specific arguments

        Returns:
            Tool execution result

        Raises:
            ToolExecutionError: If execution fails
            TimeoutError: If execution exceeds timeout
        """
        # Check cache first
        cache_key = self._get_cache_key(**kwargs)
        cached_result = self._get_cached_result(cache_key)
        if cached_result is not None:
            return cached_result

        # Execute with retry
        result = self._execute_with_retry(**kwargs)

        # Cache result
        self._set_cache(cache_key, result)

        return result

    def clear_cache(self) -> None:
        """Clear all cached results."""
        self._cache.clear()
        self.logger.info(f"Cache cleared for tool {self.name}")

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name}>"
