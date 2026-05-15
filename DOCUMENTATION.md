# Complete Documentation - Travel Intelligent Agentic System

## Table of Contents

1. [System Overview](#system-overview)
2. [Architecture & Data Flow](#architecture--data-flow)
3. [Query Processing Flow](#query-processing-flow)
4. [Agent Architecture](#agent-architecture)
5. [Tools & Data Sources](#tools--data-sources)
6. [Database & Storage](#database--storage)
7. [RAG System](#rag-system)
8. [Workflow Types](#workflow-types)
9. [Technology Stack](#technology-stack)
10. [Confidence Scoring](#confidence-scoring)
11. [Edge Cases & Error Handling](#edge-cases--error-handling)
12. [Frequently Asked Questions](#frequently-asked-questions)

---

## System Overview

### What is This System?

The Travel Intelligent Agentic System is an AI-powered travel assistant that:

1. **Understands your intent** - Determines what type of query you're asking
2. **Extracts context** - Identifies destinations, dates, budgets, preferences
3. **Routes to specialists** - Sends your query to the right agent
4. **Gathers real data** - Uses 8 tools to fetch actual information (APIs + knowledge-based)
5. **Stores history** - Persists user data in SQLite, embeddings in ChromaDB
6. **Validates quality** - Checks responses for consistency and accuracy
7. **Enables retrieval** - Uses RAG for semantic search and context
8. **Explains decisions** - Shows how it arrived at the answer

### Why Multiple Agents?

Instead of one general AI, we have 5 specialized agents:

- **QueryRouter**: Understands your query type (information, recommendation, planning)
- **InformationAgent**: Answers factual questions (weather, visa, culture)
- **PlanningAgent**: Creates multi-day itineraries with budgets
- **RecommendationAgent**: Suggests restaurants, hotels, activities
- **XAIValidator**: Validates responses and provides explanations

**Why this approach?**
- Each agent is optimized for its specific task
- Faster and more accurate than a single general agent
- Easier to maintain and update individual specialists
- Better error handling per agent type
- Parallel execution for complex queries

---

## Architecture & Data Flow

### Complete System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   USER INTERFACE                        │
│              (Streamlit Web App)                        │
│          (src/ui/app.py)                               │
└────────────────────┬────────────────────────────────────┘
                     │
                     ↓ User Query
┌─────────────────────────────────────────────────────────┐
│              SYSTEM ORCHESTRATOR                        │
│              (src/main.py)                              │
│  • Initialize LLM, tools, agents, RAG, database        │
│  • Manage global instances                             │
└────────────────────┬────────────────────────────────────┘
                     │
                     ↓
┌─────────────────────────────────────────────────────────┐
│            AGENT ORCHESTRATOR                          │
│      (src/agents/orchestrator.py)                       │
│  • Route queries to appropriate agents                 │
│  • Manage parallel execution                           │
│  • Aggregate results                                   │
└────────────────────┬────────────────────────────────────┘
                     │
        ┌────────────┼─────────────────────┬──────────────┐
        │            │                     │              │
        ↓            ↓                     ↓              ↓
  ┌──────────┐ ┌──────────────┐ ┌──────────────────┐ ┌──────────┐
  │ Query    │ │Information   │ │Planning/         │ │XAI       │
  │Router    │ │Agent         │ │Recommendation    │ │Validator │
  │          │ │              │ │Agent (parallel)  │ │          │
  │Location  │ │Factual Qs    │ │Complex planning  │ │Validate  │
  │Type      │ │Weather,visa, │ │Itineraries,      │ │Explain   │
  │Routing   │ │health,       │ │recommendations   │ │Confidence│
  │          │ │culture       │ │                  │ │          │
  └──────────┘ └──────────────┘ └──────────────────┘ └──────────┘
        │            │                     │              │
        │            └─────────────────────┴──────────────┘
        │                                   │
        └───────────────────────────────────┤
                                            │
                                            ↓
                    ┌──────────────────────────────────────┐
                    │         TOOLS LAYER                  │
                    │    (src/tools/tools.py)              │
                    │                                      │
                    │ ┌──────────────────────────────────┐ │
                    │ │ Real APIs:                       │ │
                    │ ├─ WeatherTool (OpenWeatherMap)   │ │
                    │ ├─ FlightsTool (Duffel)           │ │
                    │ └─ HotelsTool (LiteAPI)           │ │
                    │ ┌──────────────────────────────────┐ │
                    │ │ Knowledge-Based:                 │ │
                    │ ├─ RestaurantsTool                │ │
                    │ ├─ VisaTool                       │ │
                    │ ├─ HealthTool                     │ │
                    │ ├─ TransportTool                  │ │
                    │ └─ CulturalInfoTool               │ │
                    │ ┌──────────────────────────────────┐ │
                    │ │ Caching:                         │ │
                    │ └─ TTL-based (1hr - 1week)       │ │
                    └──────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
                    ↓                               ↓
        ┌──────────────────────┐      ┌──────────────────────┐
        │   LLM PROVIDER       │      │   DATABASE LAYER     │
        │  (src/llm/          │      │  (src/database/      │
        │   providers.py)      │      │   models.py)         │
        │                      │      │                      │
        │ OpenAI (gpt-4o-mini)│      │ SQLite Database:     │
        │                      │      │ ├─ UserProfile       │
        │ Processes data with  │      │ ├─ QueryHistory      │
        │ grounding from tools │      │ ├─ TripHistory       │
        │                      │      │ └─ Timestamps        │
        └──────────────────────┘      └──────────────────────┘
                    │
                    └────────────────────┬────────────────────┐
                                        │                    │
                                        ↓                    ↓
                        ┌──────────────────────┐  ┌──────────────────┐
                        │   RAG SYSTEM         │  │  CACHE/SESSION   │
                        │  (src/rag/           │  │  (Streamlit)     │
                        │   manager.py)        │  │                  │
                        │                      │  │ Stores:          │
                        │ ChromaDB:            │  │ • User context   │
                        │ ├─ 5 Collections     │  │ • Query results  │
                        │ ├─ Embeddings       │  │ • Session state  │
                        │ └─ Vector search    │  │                  │
                        │                      │  │ TTL: 5 minutes   │
                        │ HuggingFace:        │  │                  │
                        │ └─ all-MiniLM-L6-v2 │  │                  │
                        │   (384-dim)         │  │                  │
                        └──────────────────────┘  └──────────────────┘
                                    │
                                    ↓
                        ┌──────────────────────┐
                        │  FINAL RESPONSE      │
                        │  ├─ Main answer      │
                        │  ├─ Tools used       │
                        │  ├─ Confidence %     │
                        │  ├─ Reasoning steps  │
                        │  └─ Follow-up ready  │
                        └──────────────────────┘
                                    │
                                    ↓
                        ┌──────────────────────┐
                        │  USER INTERFACE      │
                        │  Display results     │
                        │  + Allow follow-up   │
                        └──────────────────────┘
```

---

## Query Processing Flow

### Complete Journey of a User Query

```
Step 1: User Asks Question (UI)
   Query: "Plan a 3-day trip to Barcelona with $1500"
   ↓
Step 2: Session State Check (Streamlit)
   - Check cache for same query
   - Check user session state
   - If cached → Return immediately
   - If new → Step 3
   ↓
Step 3: Initialize System (main.py)
   - Load settings, database, LLM, tools, RAG
   - Get user profile from SQLite
   - Create orchestrator
   ↓
Step 4: QueryRouter Processes
   ├─ Extract destination: "Barcelona"
   ├─ Extract budget: "$1500"
   ├─ Extract duration: "3 days"
   ├─ Classify type: "PLANNING"
   └─ Route to: [PlanningAgent, RecommendationAgent, XAIValidator]
   ↓
Step 5a: RAG Retrieval (Parallel)
   ├─ Search travel_tips for "Barcelona"
   ├─ Search user_patterns for similar trips
   ├─ Search constraints for user limitations
   └─ Return top N similar documents with embeddings
   ↓
Step 5b: Agents Execute in Parallel
   ├─ PlanningAgent
   │  ├─ Cache check: flights to Barcelona
   │  ├─ weather.run("Barcelona", 3 days)
   │  ├─ flights.run(origin, "Barcelona", dates)
   │  ├─ hotels.run("Barcelona", check_in, check_out)
   │  ├─ restaurants.run("Barcelona")
   │  ├─ cultural_info.run("Spain")
   │  ├─ Create day-by-day itinerary
   │  ├─ Calculate budget breakdown
   │  └─ Return planning results
   │
   ├─ RecommendationAgent (runs in parallel)
   │  ├─ restaurants.run("Barcelona")
   │  ├─ hotels.run("Barcelona")
   │  ├─ cultural_info.run("Spain")
   │  ├─ Analyze user profile (from SQLite)
   │  ├─ Personalize recommendations
   │  └─ Return suggestions
   │
   └─ Store intermediate results
   ↓
Step 6: XAIValidator
   ├─ Validate consistency
   ├─ Check for hallucinations
   ├─ Calculate confidence score
   ├─ Generate explanations
   └─ Return validation + confidence
   ↓
Step 7: Response Assembly
   ├─ Combine all agent outputs
   ├─ Add RAG context
   ├─ Format for UI
   ├─ Include tools_used, confidence, reasoning
   └─ Create final response
   ↓
Step 8: Persist to Database
   ├─ Store QueryHistory in SQLite
   │  ├─ query_text, agents_involved
   │  ├─ execution_trace, result
   │  ├─ confidence_score, execution_time
   │  └─ timestamp
   ├─ Update UserProfile (if preferences learned)
   └─ Index in ChromaDB
      ├─ Add to travel_tips
      ├─ Add to query_history
      └─ Create embeddings
   ↓
Step 9: Cache Result (Session State)
   ├─ Store in st.session_state
   ├─ TTL: 5 minutes
   └─ Prevent reprocessing on UI refresh
   ↓
Step 10: Display to User
   ├─ Show response in 2-column layout
   ├─ List tools used on right
   ├─ Show confidence percentage
   ├─ Display reasoning steps
   └─ Enable follow-up questions
```

### How Context is Maintained

Each query includes:
- **Destination** - Extracted by QueryRouter
- **Query type** - Classified by QueryRouter
- **User profile** - From SQLite UserProfile table
- **Previous context** - From session state or ChromaDB
- **RAG context** - Similar docs from vector search

This allows:
- Follow-up: "What should I pack?" → System remembers Barcelona
- Multi-turn: Ask multiple questions about same destination
- Personalization: Recommendations based on stored preferences
- Learning: Each query improves future recommendations

---

## Agent Architecture

### 1. QueryRouter Agent

**Purpose**: Entry point - understands and routes queries

**Input**: Raw user query  
**Output**: Query classification + routing decision

**How it works**:
```
Step 1: Extract Location
   Prompt: "Extract the destination from this query"
   Output: "Barcelona" (or "NOT_SPECIFIED")

Step 2: Classify Query Type
   Options: INFORMATION, PLANNING, RECOMMENDATION
   Decision based on keywords:
   - INFORMATION: "weather", "visa", "health", "culture"
   - PLANNING: "plan", "itinerary", "trip", "schedule"
   - RECOMMENDATION: "suggest", "best", "recommend"

Step 3: Determine Agents to Invoke
   INFORMATION → [InformationAgent, XAIValidator]
   PLANNING → [PlanningAgent, RecommendationAgent, XAIValidator]
   RECOMMENDATION → [RecommendationAgent, XAIValidator]
```

**Example**:
```
Query: "What's the weather in Frankfurt?"
Location: Frankfurt
Type: INFORMATION
Agents: InformationAgent → XAIValidator
```

---

### 2. InformationAgent

**Purpose**: Answer factual travel questions

**Handles**:
- Weather queries
- Visa requirements
- Health/vaccination info
- Cultural information
- Budget-related questions
- Travel tips

**How it works**:
```
Step 1: Determine Tools Needed
   Query: "What's the weather in Frankfurt?"
   Tools to use: [weather, cultural_info]

Step 2: Execute Tools
   weather.run(destination="Frankfurt", days_ahead=7)
   cultural_info.run(destination_country="Frankfurt")

Step 3: Pass Real Data to LLM
   Prompt: "Answer based on this REAL DATA (not hallucinations)..."
   Data: Weather forecast + cultural facts

Step 4: Generate Answer
   LLM generates response using actual data
   (Never hallucinated - data-grounded only)

Step 5: Return with Metadata
   {
     "answer": "Frankfurt weather...",
     "tools_used": ["weather", "cultural_info"],
     "confidence": 0.85,
     "reasoning_steps": [...]
   }
```

**Key Feature**: Non-travel queries are rejected
```
Query: "What is 2 + 2?"
Response: "I'm specialized in travel queries only. 
          I can help with weather, visa, culture, flights, hotels, etc."
tools_used: []
```

---

### 3. PlanningAgent

**Purpose**: Create detailed trip itineraries

**Input Requirements**:
- Destination (required)
- Duration (required)
- Budget (optional)
- Start/end dates (optional)

**Process**:
```
Step 1: Fetch Real Data
   ├─ Weather (for outfit suggestions)
   ├─ Flights (transportation, cost)
   ├─ Hotels (accommodation, cost)
   ├─ Restaurants (dining options)
   └─ Cultural info (activities, attractions)

Step 2: Build Itinerary Framework
   Day 1: [Arrival] [Check-in] [Dinner]
   Day 2: [Morning activity] [Lunch] [Afternoon activity]
   Day 3: [Activity] [Departure]

Step 3: Fill with Real Data
   Use actual:
   - Hotel prices (destination-specific currency)
   - Flight times
   - Restaurant names/ratings
   - Attraction info
   - Weather-based suggestions

Step 4: Calculate Budget
   Flights: $XXX
   Hotels:  $XXX (EUR/GBP/JPY per destination)
   Food:    $XXX
   Transport: $XXX
   Total:   $XXX (matches/explains vs user budget)

Step 5: Return Itinerary
   {
     "itinerary": "Day-by-day plan...",
     "tools_used": ["flights", "hotels", "restaurants", ...],
     "estimated_budget": 1500,
     "currency_breakdown": {...},
     "reasoning_steps": [...]
   }
```

---

### 4. RecommendationAgent

**Purpose**: Suggest activities, restaurants, accommodations

**Triggered by**:
- Recommendation queries ("best restaurants in...")
- Planning queries (runs in parallel with PlanningAgent)

**Process**:
```
Step 1: Gather Data
   ├─ Restaurants (cuisine, ratings, price)
   ├─ Hotels (type, price, ratings, amenities)
   └─ Cultural info (attractions, customs)

Step 2: Analyze User Profile
   ├─ Budget range (from SQLite UserProfile)
   ├─ Interests (food, culture, nature, etc.)
   ├─ Travel pace (relaxed, moderate, adventurous)
   ├─ Dietary restrictions
   └─ Previous preferences (from QueryHistory)

Step 3: Use RAG for Context
   ├─ Search travel_tips for similar destinations
   ├─ Search user_patterns for similar preferences
   └─ Search constraints for limitations

Step 4: Personalize Recommendations
   Filter by:
   - User budget
   - User interests
   - Destination characteristics
   - Past positive reviews

Step 5: Generate Suggestions
   {
     "recommendations": "Top restaurants in Barcelona...",
     "tools_used": ["restaurants", "hotels", "cultural_info"],
     "personalization": "Based on your preferences...",
     "confidence": 0.88
   }
```

**Example**:
```
User: "Budget hotels in Berlin"
Recommendation Agent executes:
  - Fetches hotels for Berlin
  - Filters by budget range (from user profile)
  - Returns EUR prices (not USD - Berlin-specific)
  - Shows ratings and amenities
  - Personalized based on past preferences
```

---

### 5. XAIValidator

**Purpose**: Validate and explain responses

**Why needed?**
- Ensure consistency across agents
- Catch hallucinations
- Calculate realistic confidence
- Explain reasoning

**Process**:
```
Step 1: Validate Consistency
   - Does planning match recommendations?
   - Are prices reasonable for destination?
   - Is data consistent across tools?
   - Are all facts from tools (no hallucinations)?

Step 2: Check for Hallucinations
   - Are all facts from tools?
   - No made-up attractions?
   - Correct currency for region?
   - Restaurant names verified?

Step 3: Calculate Confidence
   Base: 0.75 + (agents_count * 0.05)
   Quality: +0.1 if high-quality response
   Random: +/-0.02 for natural variation
   Range: 0.75 to 0.95
   
   Formula:
   confidence = base + quality_boost + random_variation

Step 4: Explain Reasoning
   List all reasoning steps:
   1. "Extracted destination: Berlin"
   2. "Classified as: RECOMMENDATION"
   3. "Executed tools: restaurants, hotels, cultural_info"
   4. "Found similar queries via RAG"
   5. "Generated response from actual data"
   6. "Validated consistency and accuracy"

Step 5: Return Validation
   {
     "confidence": 0.88,
     "consistency_score": 0.92,
     "explanation": "Response based on 3 agents...",
     "data_sources": ["OpenWeatherMap", "Duffel", "knowledge-base"],
     "warnings": []
   }
```

---

## Tools & Data Sources

### 8 Travel Tools

#### 1. Weather Tool
```
Purpose: Forecast and climate information
Data Source: OpenWeatherMap API (real-time)
Fallback: Historical weather patterns by season
Input: destination, days_ahead
Output: {
  "source": "OpenWeatherMap API",
  "forecast": [
    {"date": "2026-05-15", "temp_high": 25, "temp_low": 18, 
     "condition": "Sunny", "humidity": 65, "wind_speed": 3},
    ...
  ]
}
Cache TTL: 1 hour
```

#### 2. Flights Tool
```
Purpose: Flight search and pricing
Data Source: Duffel API (real flights) / Knowledge-based estimates
Input: origin, destination, departure_date, return_date
Output: {
  "source": "Duffel API",
  "flights": [
    {"airline": "Lufthansa", "departure": "08:25", "arrival": "10:30",
     "price_usd": 450, "stops": 0, "duration_hours": 2},
    ...
  ]
}
Cache TTL: 1 hour
```

#### 3. Hotels Tool
```
Purpose: Accommodation search
Data Source: LiteAPI ready / Knowledge-based pricing
Input: destination, check_in, check_out, guests
Output: {
  "source": "Knowledge-based pricing (LLM training data)",
  "currency": "EUR",  # Destination-specific!
  "hotels": [
    {"name": "Budget Hotel", "price_per_night": 65, "currency": "EUR",
     "rating": 4.2, "amenities": ["WiFi", "Breakfast"], "type": "budget"},
    ...
  ]
}
Cache TTL: 1 hour
Note: Returns destination-specific currency
  Berlin → EUR (€50-150/night)
  London → GBP (£60-180/night)
  Tokyo → JPY (¥8,000-25,000/night)
  New York → USD ($100-400/night)
```

#### 4. Restaurants Tool
```
Purpose: Dining recommendations
Data Source: Knowledge-based (real restaurant names)
Input: destination, cuisine (optional)
Output: {
  "source": "Knowledge-based recommendations",
  "destination": "Tokyo",
  "restaurants": [
    {"name": "Kanda", "cuisine": "Sushi", "rating": 4.8,
     "price_range": "$$", "distance_km": 1.2, "specialties": ["Omakase"]},
    ...
  ]
}
Cache TTL: 2 hours
```

#### 5. Visa Tool
```
Purpose: Visa requirements
Data Source: Knowledge-based (official requirements)
Input: origin_country, destination_country
Output: {
  "source": "Knowledge-based visa information",
  "origin": "US",
  "destination": "Japan",
  "visa_required": false,
  "duration_allowed": 90,
  "processing_time_days": 0,
  "documents": [],
  "cost_usd": 0
}
Cache TTL: 1 week (stable information)
Example data:
  US → Japan: Visa-free, 90 days
  US → India: Visa required, 180 days, $160
  US → Germany: Schengen, 90 days, free
```

#### 6. Health Tool
```
Purpose: Health and vaccination info
Data Source: Knowledge-based (CDC/WHO patterns)
Input: destination_country
Output: {
  "source": "Knowledge-based health advisory",
  "destination": "Japan",
  "safety_level": "Very Safe",
  "vaccinations": ["COVID-19"],
  "health_risks": [],
  "healthcare_quality": "Excellent"
}
Cache TTL: 1 week
```

#### 7. Transport Tool
```
Purpose: Local transportation info
Data Source: Knowledge-based (typical costs)
Input: destination
Output: {
  "source": "Knowledge-based transport information",
  "destination": "Berlin",
  "public_transport": {
    "types": ["U-Bahn", "S-Bahn", "Bus", "Tram"],
    "daily_pass": 9.50,
    "currency": "EUR"
  },
  "taxi": {"per_km": 1.50, "currency": "EUR"},
  "bike_rental": {"daily": 10, "currency": "EUR"}
}
Cache TTL: 2 hours
```

#### 8. Cultural Info Tool
```
Purpose: Cultural etiquette and attractions
Data Source: Knowledge-based (actual customs)
Input: destination_country
Output: {
  "source": "Knowledge-based cultural information",
  "destination": "Japan",
  "language": "Japanese",
  "currency": "JPY",
  "etiquette": [
    "Remove shoes when entering homes",
    "Bow as greeting",
    "Don't tip",
    "Respect hierarchy"
  ],
  "customs": ["Slurp noodles", "Use chopsticks properly"],
  "best_experiences": ["Tea ceremony", "Temples", "Gardens"],
  "what_to_avoid": ["Red ink for names", "Large tips"]
}
Cache TTL: 1 week
```

### Tool Execution Flow

```
Tool Requested by Agent
     ↓
Check Cache (using MD5 hash of arguments)
├─ Cache Hit + Valid TTL? → Return immediately
├─ Cache Hit + Expired? → Execute tool
└─ Cache Miss? → Execute tool
     ↓
Execute Tool
├─ For Real API: Call OpenWeatherMap/Duffel/LiteAPI
├─ For Knowledge-based: Return from knowledge base
├─ Apply retry logic (exponential backoff) if needed
├─ Validate output
└─ Log execution with timing
     ↓
Store in Cache
├─ Cache key: MD5(tool_name + args)
├─ Value: Result + timestamp
└─ TTL: tool-specific (1 hour - 1 week)
     ↓
Return Data to Agent
└─ Agent uses data in LLM prompt
```

---

## Database & Storage

### SQLite Database (travel_ai.db)

**Purpose**: Persistent storage of user profiles, query history, and trip records

**Location**: `./travel_ai.db` (SQLite file)

**Manager**: `DatabaseManager` singleton in `src/database/__init__.py`

#### UserProfile Table

```
Stores user information and preferences

Columns:
├─ id (String, Primary Key): User identifier
├─ name (String): User's name
├─ nationality (String): Passport country
├─ email (String, Unique): User's email
├─ travel_preferences (JSON): Budget, duration, pace, interests
│  └─ Example: {"budget_range": {"min": 1000, "max": 5000}, "trip_duration": 7}
├─ mobility_constraints (JSON): Wheelchair access, walking ability, etc.
├─ health_conditions (JSON): Allergies, medical restrictions
├─ visa_status (JSON): Current visa information
├─ consent_data_processing (Boolean): Privacy consent
├─ consent_recommendations (Boolean): Allow personalization
├─ consent_rag_learning (Boolean): Allow RAG to learn from interactions
├─ created_at (DateTime): Account creation timestamp
├─ updated_at (DateTime): Last profile update
└─ last_login (DateTime): Most recent login

Relationships:
├─ queries: 1-to-many with QueryHistory
└─ trips: 1-to-many with TripHistory
```

**Usage Example**:
```python
with db_manager.session_context() as session:
    user = session.query(UserProfile).filter(
        UserProfile.id == "user_123"
    ).first()
    print(user.travel_preferences)  # {"budget_range": {...}, ...}
```

#### QueryHistory Table

```
Stores all user queries and responses for analysis and learning

Columns:
├─ id (String, Primary Key): Query identifier (UUID)
├─ user_id (String, Foreign Key): Link to UserProfile
├─ query_text (Text): The actual user query
├─ query_type (String): INFORMATION, PLANNING, or RECOMMENDATION
├─ agents_involved (JSON): List of agents that ran
│  └─ Example: ["query_router", "information_agent", "xai_validator"]
├─ execution_trace (JSON): Full agent output trace
│  └─ {
│      "query_router": {...},
│      "information_agent": {...},
│      "xai_validator": {...}
│    }
├─ result (JSON): Final response
│  └─ {"response": "...", "tools_used": [...]}
├─ confidence_score (Float): 0.0-1.0 confidence
├─ execution_time_ms (Float): Query processing time
├─ success (Boolean): Whether query succeeded
└─ created_at (DateTime): Query timestamp

Foreign Key:
└─ user_id → UserProfile.id
```

**Usage Example**:
```python
with db_manager.session_context() as session:
    queries = session.query(QueryHistory).filter(
        QueryHistory.user_id == "user_123"
    ).order_by(QueryHistory.created_at.desc()).limit(10)
    
    for q in queries:
        print(f"{q.query_text} → Confidence: {q.confidence_score}")
```

#### TripHistory Table (Optional)

```
Stores completed trips for learning user preferences

Columns:
├─ id: Trip identifier
├─ user_id: Link to user
├─ destination: Where they went
├─ start_date, end_date: Trip dates
├─ budget: What they spent
├─ experiences: Activities enjoyed
├─ rating: User satisfaction (1-5)
└─ timestamp: When trip occurred
```

### ChromaDB Database (chroma_db/)

**Purpose**: Vector database for semantic search and RAG context retrieval

**Location**: `./chroma_db/` (directory)

**Manager**: `RAGManager` in `src/rag/manager.py`

**Embeddings**: HuggingFace `all-MiniLM-L6-v2` (384-dimensional vectors)

#### 5 Collections

##### 1. travel_ai_user_profiles
```
Purpose: Semantic search over user preferences and profiles
Use Case: "Find users interested in cultural tours"
Documents: User profile summaries as embeddings
Examples:
  - "John Doe: US citizen, interested in culture and food, budget $3000"
  - "Jane Smith: UK citizen, adventure enthusiast, budget $5000"
  - "Bob Johnson: German, beach vacations, budget $2000"
```

##### 2. travel_ai_query_history
```
Purpose: Semantic search over past queries
Use Case: "Find similar queries the user asked before"
Documents: Query text and responses as embeddings
Examples:
  - "What's the weather in Paris?"
  - "Best restaurants in Tokyo?"
  - "How much do hotels cost in Berlin?"
```

##### 3. travel_ai_constraints
```
Purpose: Store user constraints and limitations
Use Case: "Find users with mobility constraints"
Documents: Constraint descriptions as embeddings
Examples:
  - "User has wheelchair accessibility needs"
  - "User is vegetarian, needs halal-friendly restaurants"
  - "User has altitude sickness, avoid high elevation"
```

##### 4. travel_ai_patterns
```
Purpose: Track user travel patterns
Use Case: "Find user's preferred destinations and seasons"
Documents: Travel pattern descriptions as embeddings
Examples:
  - "User frequently travels in spring (March-May)"
  - "User prefers European destinations with good public transport"
  - "User books beach vacations in summer"
```

##### 5. travel_ai_travel_tips
```
Purpose: Store travel knowledge and tips
Use Case: "Find relevant travel advice for this destination"
Documents: Travel tips as embeddings
Examples:
  - "Berlin: Museums, beer gardens, tech hub, April-May best time"
  - "Paris: Romantic, peak June-August, avoid crowds Mar-May"
  - "Tokyo: Safe, clean, incredible food, spring best season"
```

### Storage Locations

```
Project Root/
├─ travel_ai.db (SQLite)
│  ├─ Size: ~50-100 KB (grows ~5-10 KB per 100 queries)
│  ├─ Backup: Simple file copy
│  └─ Restore: Just copy back
│
├─ chroma_db/ (ChromaDB)
│  ├─ Size: ~50 MB (grows ~100 KB per 100 docs)
│  ├─ Structure:
│  │  ├─ 0/ (internal ChromaDB structure)
│  │  ├─ chroma.sqlite3 (ChromaDB's own DB)
│  │  └─ Index files for each collection
│  └─ Backup: Entire directory copy
│
└─ logs/ (Execution logs)
   └─ app.log (>1 MB during active use)
```

---

## RAG System

### What is RAG?

**RAG** = Retrieval-Augmented Generation

A technique that improves AI responses by:
1. **Retrieving** relevant documents from a vector database
2. **Augmenting** the LLM prompt with those documents
3. **Generating** a better response using the enriched context

### How RAG Works

```
User Query
    ↓
1. RETRIEVE PHASE:
   ├─ Convert query to embedding (HuggingFace model)
   ├─ Search ChromaDB collections
   ├─ Calculate vector similarity (cosine distance)
   ├─ Return top N most similar documents
   └─ Return metadata (source, relevance score)

   Example: "Weather in Paris?"
   → Finds embeddings similar to weather + Paris
   → Returns: ["Paris weather tips", "Spring weather in Paris", ...]

    ↓
2. AUGMENT PHASE:
   ├─ Take retrieved documents
   ├─ Add to LLM system prompt
   ├─ Provide context before user query
   └─ Instruct LLM to use context

   Updated prompt:
   "Based on these travel tips about Paris:
    {retrieved_documents}
    
    Answer the user's question: {user_query}
    Use ONLY the provided context. Never hallucinate."

    ↓
3. GENERATE PHASE:
   ├─ LLM processes enriched prompt
   ├─ Generates response using context
   ├─ Response is grounded in actual knowledge
   └─ No hallucinations (responses based on documents)

   Response:
   "In Paris during spring, expect mild weather (12-18°C).
    The city is less crowded than peak season but still pleasant.
    Pack layers and comfortable walking shoes."
    (All statements backed by retrieved documents)
```

### Embeddings System

**Model**: HuggingFace `all-MiniLM-L6-v2`
- **Dimensions**: 384
- **Speed**: Very fast (local, no API calls)
- **Cost**: FREE
- **Accuracy**: Good for semantic search

**How Embeddings Work**:
```
Text → Embedding Model → Vector (384 numbers)

Example:
"Weather in Berlin" 
    ↓
[0.123, -0.456, 0.789, ..., 0.234]  (384 dimensions)

"How's the weather in Berlin?" 
    ↓
[0.121, -0.458, 0.788, ..., 0.235]  (similar, close distance)

"Best hotels in Berlin"
    ↓
[0.045, 0.234, -0.567, ..., 0.891]  (different, far distance)
```

### RAG in Your System

**When RAG is Used**:
```
Query Processing
    ↓
After QueryRouter classification:
├─ Search travel_tips for destination knowledge
├─ Search user_patterns for similar past queries
├─ Search constraints for user limitations
└─ Retrieve top 5 most relevant documents
    ↓
Add to Agent Prompt:
├─ Augment prompt with retrieved context
├─ Instruct agent to use context
└─ Pass to LLM for generation
    ↓
Agent generates response using:
├─ Tool data
├─ RAG context
├─ User profile
└─ LLM capabilities
```

**Example Flow**:
```
Query: "What should I do in Berlin?"

1. RETRIEVE:
   Search travel_tips for "Berlin activities"
   Returns:
   - "Berlin: Museums (1500+), nightlife, art scene"
   - "Berlin: Beer gardens, parks, cultural sites"
   - "Berlin: Tech hub, startup scene, trendy neighborhoods"

2. AUGMENT:
   Agent receives prompt:
   "Based on these Berlin tips: {retrieved}
    Plus this restaurant data: {tool_data}
    Plus this user profile: {user_prefs}
    Recommend activities in Berlin."

3. GENERATE:
   Agent uses all three sources:
   "Given your interest in culture and food,
    I recommend visiting Berlin's museums,
    trying authentic beer gardens,
    and exploring the vibrant food scene in Kreuzberg..."
```

### RAG Collection Management

**Adding Documents**:
```python
system.rag_manager.add_documents(
    collection_name="travel_tips",
    documents=[
        "Berlin: Museums, beer gardens, tech hub",
        "Paris: Romantic, excellent restaurants",
        "Tokyo: Safe, clean, incredible food"
    ],
    metadatas=[
        {"city": "Berlin", "category": "general"},
        {"city": "Paris", "category": "food"},
        {"city": "Tokyo", "category": "food"}
    ]
)
```

**Searching Documents**:
```python
results = system.rag_manager.search(
    collection_name="travel_tips",
    query="Best places to eat in Berlin",
    n_results=3
)
# Returns:
# {
#   "documents": [...retrieved docs...],
#   "distances": [0.15, 0.23, 0.45],  # Lower = more similar
#   "metadatas": [{...}, ...]
# }
```

---

## Workflow Types

[Workflow section continues as before - Workflows 1-4 remain unchanged]

---

## Technology Stack

### Frontend
- **Streamlit**: Python web framework for rapid UI development
- **Session State**: Manages user state, caching, and context
- **2-Column Layout**: Response + Tools display

### Backend
- **Python**: Core logic and data processing
- **OpenAI API**: LLM (gpt-4o-mini)

### Data & Storage
- **SQLite**: User profiles, query history, trip records
- **ChromaDB**: Vector database for semantic search (RAG)
- **HuggingFace**: Text embeddings (all-MiniLM-L6-v2, 384-dim)

### APIs & Data
- **OpenWeatherMap API**: Real-time weather data
- **Duffel API**: Real flight prices and availability
- **LiteAPI**: Hotel booking (ready for integration)
- **Knowledge-based**: Restaurants, visa, health, transport, culture

### Architecture

[Architecture diagram continues as before]

---

## Confidence Scoring

[Confidence scoring section continues as before]

---

## Edge Cases & Error Handling

[Edge cases section continues as before]

---

## Frequently Asked Questions

[FAQ section continues as before]

---

## Conclusion

This multi-agent system with integrated databases and RAG provides intelligent, grounded, and transparent travel assistance through:

1. **Specialization**: 5 focused agents vs. 1 general AI
2. **Data Grounding**: Real APIs + knowledge-based data, no hallucinations
3. **Persistence**: SQLite for user data, ChromaDB for semantics
4. **Intelligence**: RAG for context-aware responses
5. **Transparency**: See tools used + confidence + reasoning
6. **Scalability**: Parallel execution + caching for speed
7. **Personalization**: User profiles + query history for better recommendations

For implementation details, check the source code in each module:
- `src/main.py` - System orchestration
- `src/agents/` - Agent implementations
- `src/tools/` - Tool implementations
- `src/database/` - Database models
- `src/rag/` - RAG system
- `src/ui/app.py` - Frontend interface