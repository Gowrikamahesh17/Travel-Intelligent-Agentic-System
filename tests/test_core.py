"""
Basic tests for Travel Intelligent Agentic System.
Run with: pytest tests/
"""

import pytest
from unittest.mock import MagicMock, patch
from src.common import (
    ConfigurationError,
    AgentExecutionError,
    ToolExecutionError,
    get_settings,
)
from src.tools import WeatherTool, VisaTool, FlightsTool
from src.agents import QueryRouter


@pytest.fixture
def mock_llm():
    """Create mock LLM."""
    llm = MagicMock()
    llm.generate = MagicMock(return_value="Test response")
    return llm


class TestExceptions:
    """Test custom exceptions."""

    def test_configuration_error(self):
        """Test ConfigurationError."""
        error = ConfigurationError("Missing API key")
        assert error.error_code == "CONFIG_ERROR"
        assert error.retryable is False

    def test_agent_execution_error(self):
        """Test AgentExecutionError."""
        error = AgentExecutionError(
            "Agent failed", agent_name="TestAgent", retryable=True
        )
        assert error.error_code == "AGENT_ERROR"
        assert error.context["agent_name"] == "TestAgent"
        assert error.retryable is True

    def test_tool_execution_error(self):
        """Test ToolExecutionError."""
        error = ToolExecutionError("Tool failed", tool_name="TestTool")
        assert error.context["tool_name"] == "TestTool"


class TestTools:
    """Test tool implementations."""

    def test_weather_tool(self):
        """Test WeatherTool."""
        tool = WeatherTool()
        result = tool.run(destination="Tokyo", days_ahead=3)

        assert result["destination"] == "Tokyo"
        assert len(result["forecast"]) == 3
        assert "temp_high" in result["forecast"][0]

    def test_visa_tool(self):
        """Test VisaTool."""
        tool = VisaTool()
        result = tool.run(origin_country="US", destination_country="Japan")

        assert result["origin"] == "US"
        assert result["destination"] == "Japan"
        assert "visa_required" in result

    def test_flights_tool(self):
        """Test FlightsTool."""
        tool = FlightsTool()
        result = tool.run(
            origin="SFO",
            destination="NRT",
            departure_date="2024-06-01",
        )

        assert result["origin"] == "SFO"
        assert result["destination"] == "NRT"
        assert len(result["flights"]) > 0

    def test_tool_caching(self):
        """Test tool result caching."""
        tool = WeatherTool()

        # First call
        result1 = tool.run(destination="Tokyo")

        # Second call should be from cache
        result2 = tool.run(destination="Tokyo")

        assert result1 == result2

    def test_tool_cache_clear(self):
        """Test cache clearing."""
        tool = WeatherTool()
        tool.run(destination="Tokyo")
        tool.clear_cache()

        assert len(tool._cache) == 0


class TestAgents:
    """Test agent implementations."""

    def test_query_router_initialization(self, mock_llm):
        """Test QueryRouter initialization."""
        router = QueryRouter(mock_llm)

        assert router.name == "QueryRouter"
        assert router.llm == mock_llm

    def test_query_router_execution(self, mock_llm):
        """Test QueryRouter execution."""
        mock_llm.generate = MagicMock(return_value="PLANNING")
        router = QueryRouter(mock_llm)

        result = router.execute(query="Plan a trip to Japan")

        assert "query_type" in result
        assert "agents_to_invoke" in result
        assert result["query"] == "Plan a trip to Japan"


class TestSettings:
    """Test settings management."""

    def test_settings_loading(self):
        """Test settings loading."""
        with patch.dict("os.environ", {"PRIMARY_LLM_PROVIDER": "gemini"}):
            from src.common import reload_settings

            settings = reload_settings()
            assert settings.PRIMARY_LLM_PROVIDER == "gemini"

    def test_settings_defaults(self):
        """Test settings defaults."""
        settings = get_settings()

        assert settings.CACHE_TTL_SECONDS == 86400
        assert settings.API_TIMEOUT == 30
        assert settings.ENABLE_RAG is True


class TestToolRegistry:
    """Test tool registry."""

    def test_get_all_tools(self):
        """Test getting all tools."""
        from src.tools import get_all_tools

        tools = get_all_tools()

        assert "weather" in tools
        assert "flights" in tools
        assert "visa" in tools
        assert len(tools) == 8

    def test_get_specific_tool(self):
        """Test getting specific tool."""
        from src.tools import get_tool

        tool = get_tool("weather")
        assert tool.name == "weather"

    def test_get_invalid_tool(self):
        """Test getting invalid tool."""
        from src.tools import get_tool

        with pytest.raises(ValueError):
            get_tool("invalid_tool")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
