"""
Example script: Basic usage of Travel Intelligent Agentic System
Shows how to initialize and use the system programmatically.
"""

from src.main import initialize_system
import json


def example_1_basic_query():
    """Example 1: Process a basic travel planning query."""
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic Travel Planning Query")
    print("="*60)

    system = initialize_system()

    query = "Plan a 5-day trip to Tokyo with a $2000 budget"
    print(f"\nQuery: {query}")

    result = system.process_query(
        query=query,
        user_id="example_user_1",
        context={
            "destination": "Tokyo",
            "start_date": "2024-06-01",
            "end_date": "2024-06-05",
            "budget": 2000,
        },
    )

    print(f"\nResponse:\n{result['response'][:300]}...")
    print(f"\nExecution Time: {result['execution_time_ms']:.2f}ms")
    print(f"Validation: {result['validation']['validation_status']}")


def example_2_information_query():
    """Example 2: Process an information query."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Information Query (Visa Requirements)")
    print("="*60)

    system = initialize_system()

    query = "What are the visa requirements for US citizens traveling to Thailand?"
    print(f"\nQuery: {query}")

    result = system.process_query(
        query=query,
        user_id="example_user_2",
        context={
            "origin": "US",
            "destination": "Thailand",
        },
    )

    print(f"\nResponse:\n{result['response'][:300]}...")
    print(f"\nExecution Time: {result['execution_time_ms']:.2f}ms")


def example_3_user_profile():
    """Example 3: Create user profile and use personalization."""
    print("\n" + "="*60)
    print("EXAMPLE 3: User Profile & Personalization")
    print("="*60)

    system = initialize_system()

    # Create user profile
    user_id = "example_user_3"
    user = system.create_user_profile(
        user_id=user_id,
        name="Alice Johnson",
        nationality="US",
        email="alice@example.com",
        preferences={
            "budget_range": {"min": 1500, "max": 4000},
            "trip_duration": 7,
            "travel_pace": "moderate",
            "interests": ["culture", "food", "nature"],
        },
    )

    print(f"\nCreated user: {user.name}")
    print(f"Preferences: {user.travel_preferences}")

    # Process query with user context
    query = "Recommend a 7-day itinerary for Europe"
    print(f"\nQuery: {query}")

    result = system.process_query(
        query=query,
        user_id=user_id,
        context={
            "destination": "Europe",
            "start_date": "2024-07-01",
            "end_date": "2024-07-07",
            "budget": 3000,
        },
    )

    print(f"\nResponse:\n{result['response'][:300]}...")


def example_4_agent_trace():
    """Example 4: Inspect agent execution trace."""
    print("\n" + "="*60)
    print("EXAMPLE 4: Agent Execution Trace")
    print("="*60)

    system = initialize_system()

    query = "What's the best time to visit Bali?"
    result = system.process_query(query=query, user_id="example_user_4")

    print(f"\nQuery: {query}")
    print("\nAgent Execution Trace:")

    for agent_name, agent_output in result["agent_trace"].items():
        print(f"\n{agent_name.upper()}:")
        print(f"  - Status: Success")
        if "query_type" in agent_output:
            print(f"  - Query Type: {agent_output['query_type']}")
        if "reasoning_steps" in agent_output:
            print(f"  - Reasoning Steps: {agent_output['reasoning_steps']}")


def example_5_tool_usage():
    """Example 5: Using individual tools."""
    print("\n" + "="*60)
    print("EXAMPLE 5: Using Individual Tools")
    print("="*60)

    system = initialize_system()

    # Get weather tool
    weather_tool = system.get_tool("weather")
    if weather_tool:
        print("\nFetching weather for Tokyo...")
        weather = weather_tool.run(destination="Tokyo", days_ahead=7)
        print(f"Forecast: {weather['forecast'][0]}")

    # Get visa tool
    visa_tool = system.get_tool("visa")
    if visa_tool:
        print("\nFetching visa info...")
        visa_info = visa_tool.run(origin_country="US", destination_country="Japan")
        print(f"Visa Required: {visa_info['visa_required']}")
        print(f"Processing Time: {visa_info['processing_time_days']} days")

    # Get flights tool
    flights_tool = system.get_tool("flights")
    if flights_tool:
        print("\nSearching flights...")
        flights = flights_tool.run(
            origin="SFO",
            destination="NRT",
            departure_date="2024-06-01",
        )
        print(f"Found {len(flights['flights'])} flights")
        print(f"Cheapest: ${flights['cheapest_price']}")


def example_6_system_status():
    """Example 6: Check system status."""
    print("\n" + "="*60)
    print("EXAMPLE 6: System Status")
    print("="*60)

    system = initialize_system()
    status = system.get_system_status()

    print("\nSystem Status:")
    print(json.dumps(status, indent=2))


def main():
    """Run all examples."""
    print("\n🌍 Travel Intelligent Agentic System - Examples")
    print("=" * 60)

    try:
        # Run examples
        example_1_basic_query()
        example_2_information_query()
        example_3_user_profile()
        example_4_agent_trace()
        example_5_tool_usage()
        example_6_system_status()

        print("\n" + "="*60)
        print("✅ All examples completed successfully!")
        print("="*60)

    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
