"""
Streamlit UI — Travel Intelligent Agentic System.
"""

import os
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import streamlit as st
import json
import sys
import logging
import hashlib
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List

logging.getLogger("transformers").setLevel(logging.ERROR)

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from src.main import process_query as _process_query
    from src.common import get_logger
    logger = get_logger(__name__)
except ImportError as e:
    logger = None
    import_error = str(e)

# ── All 8 tools with metadata ────────────────────────────────────────────────
ALL_TOOLS = {
    "weather":      {"label": "Weather",       "icon": "🌤️", "api": "OpenWeatherMap / Open-Meteo"},
    "flights":      {"label": "Flights",        "icon": "✈️", "api": "Duffel API"},
    "hotels":       {"label": "Hotels",         "icon": "🏨", "api": "LLM Knowledge"},
    "restaurants":  {"label": "Restaurants",    "icon": "🍽️", "api": "OpenStreetMap / LLM"},
    "visa":         {"label": "Visa",           "icon": "🛂", "api": "REST Countries / LLM"},
    "health":       {"label": "Health",         "icon": "🏥", "api": "LLM Knowledge"},
    "transport":    {"label": "Transport",      "icon": "🚌", "api": "LLM Knowledge"},
    "cultural_info":{"label": "Culture",        "icon": "🏛️", "api": "REST Countries / LLM"},
}

ALL_AGENTS = {
    "query_router":         {"label": "Query Router",       "icon": "🔀", "tools": []},
    "information_agent":    {"label": "Information Agent",  "icon": "ℹ️",  "tools": list(ALL_TOOLS.keys())},
    "planning_agent":       {"label": "Planning Agent",     "icon": "🗓️", "tools": ["weather","flights","hotels","cultural_info"]},
    "recommendation_agent": {"label": "Recommendation Agent","icon": "⭐", "tools": ["restaurants","hotels","cultural_info"]},
    "xai_validator":        {"label": "XAI Validator",      "icon": "✅", "tools": []},
}


# ── Helpers ──────────────────────────────────────────────────────────────────

def _call_backend(query: str, destination: Optional[str] = None,
                  budget: Optional[int] = None,
                  user_profile: Optional[Dict] = None,
                  context: Optional[Dict] = None) -> Dict[str, Any]:
    """Call backend with per-query caching."""
    key = hashlib.md5((query + str(destination or "") + str(budget or "")).encode()).hexdigest()
    st.session_state.setdefault("query_cache", {})
    if key in st.session_state.query_cache:
        return st.session_state.query_cache[key]
    result = _process_query(query=query, destination=destination, budget=budget,
                            user_profile=user_profile, context=context or {})
    st.session_state.query_cache[key] = result
    return result


def _get_main_answer(result: Dict[str, Any]) -> str:
    trace = result.get("agent_trace", {})
    for key, field in [("information_agent", "answer"),
                       ("planning_agent", "itinerary"),
                       ("recommendation_agent", "recommendations")]:
        data = trace.get(key, {})
        if isinstance(data, dict) and data.get(field):
            return data[field]
    return result.get("response", "No response generated.")


def _get_tool_data(result: Dict[str, Any]) -> Dict:
    """Collect tool_data from all agent outputs (including indexed sub-query agents)."""
    out = {}
    trace = result.get("agent_trace", {})
    for key, agent in trace.items():
        if isinstance(agent, dict) and "tool_data" in agent:
            out.update(agent["tool_data"])
    return out


def _parse_agent_tool_usage(result: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    Return {canonical_agent_key: [tool_name, ...]} for agents that actually ran.
    Handles both simple keys ('information_agent') and indexed sub-query keys
    ('information_agent_0', 'information_agent_1') from LangGraph multi-intent.
    """
    usage: Dict[str, List[str]] = {}
    trace = result.get("agent_trace", {})

    # Canonical keys (always shown in panel even if skipped)
    canonical = ("query_router", "information_agent", "planning_agent",
                 "recommendation_agent", "xai_validator")

    for key, agent_data in trace.items():
        if not isinstance(agent_data, dict):
            continue
        # Map indexed keys back to canonical: "information_agent_1" → "information_agent"
        canonical_key = key
        for c in canonical:
            if key == c or key.startswith(c + "_"):
                canonical_key = c
                break
        tools = agent_data.get("tools_used", [])
        if canonical_key in usage:
            # Merge tools from multiple sub-queries for the same agent type
            for t in tools:
                if t not in usage[canonical_key]:
                    usage[canonical_key].append(t)
        else:
            usage[canonical_key] = list(tools)

    # Always include query_router and xai_validator if present
    for key in ("query_router", "xai_validator"):
        if key in trace and key not in usage:
            usage[key] = []

    return usage


def _live_source(tool_data_entry: Dict) -> bool:
    """True if tool returned real API data (not llm_knowledge signal)."""
    src = tool_data_entry.get("source", "") if isinstance(tool_data_entry, dict) else ""
    return "llm_knowledge" not in src


# ── Agent-Tool hierarchy panel ────────────────────────────────────────────────

def render_agent_tool_panel(result: Dict[str, Any]) -> None:
    """Right-panel: hierarchy of agents invoked and tools used / available."""

    agent_tool_usage = _parse_agent_tool_usage(result)
    tool_data = _get_tool_data(result)

    # Collect all tools actually used across all agents
    all_used_tools: set = set()
    for tools in agent_tool_usage.values():
        all_used_tools.update(tools)

    st.markdown(
        """
        <style>
        .panel-title {font-size:11px;font-weight:700;color:#6b7280;
                      text-transform:uppercase;letter-spacing:.08em;margin-bottom:6px;}
        .agent-box {border-radius:6px;padding:6px 8px;margin-bottom:5px;
                    background:#f8fafc;border:1px solid #e2e8f0;}
        .agent-active {background:#eff6ff;border-color:#93c5fd;}
        .agent-name {font-size:11px;font-weight:700;color:#1e3a5f;
                     display:flex;align-items:center;justify-content:space-between;
                     flex-wrap:wrap;gap:2px;}
        .tool-row {display:flex;align-items:center;justify-content:space-between;
                   font-size:10px;padding:2px 0 0 10px;color:#374151;line-height:1.4;}
        .tool-name {flex:1;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
        .badge-live {background:#dcfce7;color:#15803d;border-radius:3px;
                     padding:0px 4px;font-size:9px;font-weight:700;white-space:nowrap;flex-shrink:0;}
        .badge-llm  {background:#fef9c3;color:#854d0e;border-radius:3px;
                     padding:0px 4px;font-size:9px;font-weight:700;white-space:nowrap;flex-shrink:0;}
        .badge-skip {background:#f3f4f6;color:#9ca3af;border-radius:3px;
                     padding:0px 4px;font-size:9px;white-space:nowrap;flex-shrink:0;}
        .dot-used  {width:6px;height:6px;border-radius:50%;min-width:6px;
                    background:#22c55e;display:inline-block;margin-right:4px;}
        .dot-skip  {width:6px;height:6px;border-radius:50%;min-width:6px;
                    background:#d1d5db;display:inline-block;margin-right:4px;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="panel-title">Agent & Tool Execution</div>', unsafe_allow_html=True)

    for agent_key, agent_meta in ALL_AGENTS.items():
        was_invoked = agent_key in agent_tool_usage
        tools_run = agent_tool_usage.get(agent_key, [])

        box_class = "agent-box agent-active" if was_invoked else "agent-box"
        invoked_badge = (
            "<span style='background:#bbf7d0;color:#166534;border-radius:4px;"
            "padding:1px 6px;font-size:10px;font-weight:700;float:right'>INVOKED</span>"
            if was_invoked
            else "<span style='background:#f3f4f6;color:#9ca3af;border-radius:4px;"
            "padding:1px 6px;font-size:10px;float:right'>SKIPPED</span>"
        )

        tool_html = ""
        for tool_key in agent_meta["tools"]:
            tmeta = ALL_TOOLS[tool_key]
            if tool_key in tools_run:
                td = tool_data.get(tool_key, {})
                if isinstance(td, dict) and _live_source(td):
                    badge = '<span class="badge-live">LIVE API</span>'
                else:
                    badge = '<span class="badge-llm">LLM KNOWLEDGE</span>'
                dot = '<span class="dot-used"></span>'
            else:
                badge = '<span class="badge-skip">NOT USED</span>'
                dot = '<span class="dot-skip"></span>'

            tool_html += (
                f'<div class="tool-row">'
                f'{dot}<span class="tool-name">{tmeta["icon"]} {tmeta["label"]}</span>{badge}'
                f'</div>'
            )

        st.markdown(
            f'<div class="{box_class}">'
            f'<div class="agent-name">{agent_meta["icon"]} {agent_meta["label"]}{invoked_badge}</div>'
            f"{tool_html}"
            f"</div>",
            unsafe_allow_html=True,
        )

    # Confidence
    confidence = result.get("validation", {}).get("confidence_score", 0.85)
    live_n = result.get("validation", {}).get("live_data_tools", 0)
    total_n = result.get("validation", {}).get("total_tools", 0)
    st.markdown(
        f"""
        <div style="margin-top:10px;padding:8px 10px;border-radius:8px;
             background:#f8fafc;border:1px solid #e2e8f0;font-size:12px;">
          <span style="color:#6b7280;">Confidence</span>
          <span style="font-weight:700;color:#1e3a5f;float:right">{confidence:.0%}</span><br>
          <span style="color:#6b7280;">Live API tools</span>
          <span style="font-weight:600;color:#15803d;float:right">{live_n} / {total_n}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Live data expanders ───────────────────────────────────────────────────────

def render_live_data(tool_data: Dict[str, Any]) -> None:
    """Render expandable sections for tools that returned live API data."""
    if not tool_data:
        return

    rendered = False

    weather = tool_data.get("weather", {})
    if isinstance(weather, dict) and _live_source(weather) and weather.get("current"):
        rendered = True
        cur = weather["current"]
        fc = weather.get("forecast", [])
        with st.expander("🌤️ Live Weather", expanded=True):
            c1, c2, c3 = st.columns(3)
            c1.metric("Temp", f"{cur.get('temperature', '?')}°C",
                      delta=f"feels {cur.get('feels_like', '?')}°C")
            c2.metric("Condition", cur.get("condition", "?"))
            c3.metric("Wind", f"{cur.get('wind_speed', '?')} m/s")
            if fc:
                st.markdown("**7-day forecast**")
                for day in fc[:7]:
                    d = str(day.get("date", ""))[:10]
                    hi = day.get("temp_high", "?")
                    lo = day.get("temp_low", "?")
                    cond = day.get("condition", "?")
                    prec = day.get("precipitation_probability")
                    prec_str = f"  💧{prec}%" if prec is not None else ""
                    st.markdown(f"- **{d}** · {cond} · {lo}–{hi}°C{prec_str}")
            st.caption(f"Source: {weather.get('source', '')}")

    restaurants = tool_data.get("restaurants", {})
    if (isinstance(restaurants, dict) and _live_source(restaurants)
            and restaurants.get("restaurants")):
        rendered = True
        with st.expander("🍽️ Restaurants (OpenStreetMap)", expanded=False):
            for r in restaurants["restaurants"][:10]:
                name = r.get("name", "?")
                cuisine = r.get("cuisine", "")
                addr = r.get("address", "")
                oh = r.get("opening_hours", "")
                detail = " · ".join(filter(None, [cuisine, addr, oh]))
                st.markdown(f"- **{name}**" + (f"  _{detail}_" if detail else ""))
            st.caption(f"Source: {restaurants.get('source', '')}")

    flights = tool_data.get("flights", {})
    if isinstance(flights, dict) and _live_source(flights) and flights.get("flights"):
        rendered = True
        with st.expander("✈️ Live Flights", expanded=False):
            for f in flights["flights"][:5]:
                st.markdown(
                    f"- **{f.get('airline','?')}** {f.get('flight_number','')} · "
                    f"{f.get('departure_time','?')} → {f.get('arrival_time','?')} · "
                    f"{f.get('stops',0)} stop(s) · "
                    f"{f.get('currency','USD')} {f.get('price_usd','?')}"
                )
            st.caption(f"Source: {flights.get('source', '')}")

    hotels = tool_data.get("hotels", {})
    if isinstance(hotels, dict) and _live_source(hotels) and hotels.get("hotels"):
        rendered = True
        with st.expander("🏨 Hotels (OpenStreetMap)", expanded=False):
            for h in hotels["hotels"][:10]:
                stars = ""
                if h.get("stars"):
                    try:
                        stars = " ★" * int(h["stars"])
                    except Exception:
                        stars = f" {h['stars']}★"
                htype = h.get("type", "hotel").replace("_", " ").title()
                addr = h.get("address", "")
                website = h.get("website", "")
                detail = " · ".join(filter(None, [htype, addr]))
                link = f" [🔗]({website})" if website else ""
                st.markdown(f"- **{h.get('name','?')}**{stars}{link}  _{detail}_" if detail else f"- **{h.get('name','?')}**{stars}{link}")
            st.caption(f"Source: {hotels.get('source','')} · Prices not available — check Booking.com")

    visa = tool_data.get("visa", {})
    if isinstance(visa, dict) and visa.get("country_info"):
        rendered = True
        ci = visa["country_info"]
        with st.expander("🌍 Destination Facts", expanded=False):
            cols = st.columns(2)
            cols[0].markdown(f"**Capital:** {ci.get('capital','?')}")
            cols[0].markdown(f"**Currency:** {ci.get('currency','?')}")
            cols[1].markdown(f"**Region:** {ci.get('region','?')}")
            langs = ci.get("languages", [])
            if langs:
                cols[1].markdown(f"**Language(s):** {', '.join(langs)}")
            st.caption("Source: REST Countries API")


# ── Follow-up result block ────────────────────────────────────────────────────

def _render_followup_block(fu_result: Dict, fu_query: str) -> None:
    st.markdown(f"---\n**Follow-up:** _{fu_query}_")
    fu_col, fu_panel = st.columns([3, 1], gap="large")
    with fu_col:
        st.markdown(_get_main_answer(fu_result))
        render_live_data(_get_tool_data(fu_result))
    with fu_panel:
        render_agent_tool_panel(fu_result)


# ── Result render ─────────────────────────────────────────────────────────────

def _render_result(result: Dict[str, Any], query: str) -> None:
    st.markdown("---")

    answer_tab, trace_tab = st.tabs(["💬 Answer", "🔍 Agent Trace (JSON)"])

    with answer_tab:
        # Main 3:1 split — answer left, agent panel right
        col_answer, col_panel = st.columns([2.5, 1], gap="medium")

        with col_answer:
            st.markdown(_get_main_answer(result))

            tool_data = _get_tool_data(result)
            if tool_data:
                render_live_data(tool_data)

        with col_panel:
            render_agent_tool_panel(result)

        # ── Follow-up ──────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown(
            "<p style='font-size:13px;color:#64748b;margin-bottom:6px'>💬 Ask a follow-up question</p>",
            unsafe_allow_html=True,
        )
        fu_col_input, fu_col_btn = st.columns([5, 1], gap="small")
        with fu_col_input:
            follow_up = st.text_input(
                "follow_up",
                placeholder="e.g. Is it going to rain this week? / Best restaurants nearby?",
                key="follow_up_input",
                label_visibility="collapsed",
            )
        with fu_col_btn:
            # Use label as vertical spacer so button aligns with input
            submit_fu = st.button("Ask ↩", use_container_width=True, key="followup_button",
                                  type="secondary", help="Ask a follow-up about the same destination")

        if submit_fu and follow_up:
            with st.spinner("Thinking…"):
                try:
                    loc = result.get("agent_trace", {}).get("query_router", {}).get("location")
                    fu_res = _call_backend(
                        query=follow_up,
                        destination=loc,
                        user_profile=st.session_state.user_profile,
                        context={"previous_query": query},
                    )
                    st.session_state.followup_result = fu_res
                    st.session_state.followup_query = follow_up
                except Exception as e:
                    st.error(f"Follow-up failed: {e}")

        if st.session_state.get("followup_result"):
            _render_followup_block(
                st.session_state.followup_result,
                st.session_state.followup_query,
            )

    with trace_tab:
        router = result.get("agent_trace", {}).get("query_router", {})
        st.json({
            "query_type": router.get("query_type"),
            "location": router.get("location"),
            "tools_needed": router.get("tools_needed"),
            "confidence": result.get("validation", {}).get("confidence_score"),
            "execution_time_ms": round(result.get("execution_time_ms", 0), 1),
            "agents_invoked": list(_parse_agent_tool_usage(result).keys()),
        })


# ── Session state ─────────────────────────────────────────────────────────────

def _init_session():
    defaults = {
        "user_profile": {
            "name": "Traveller",
            "nationality": "US",
            "email": "",
            "travel_preferences": {
                "budget_range": {"min": 1000, "max": 5000},
                "trip_duration": 7,
                "travel_pace": "moderate",
                "interests": ["culture", "food"],
            },
            "dietary_restrictions": [],
            "mobility_needs": [],
        },
        "query_history": [],
        "last_result": None,
        "last_query": None,
        "followup_result": None,
        "followup_query": None,
        "query_cache": {},
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


# ── Tabs ──────────────────────────────────────────────────────────────────────

def render_query_tab():
    # Tight, clean query box
    st.markdown("#### What would you like to know?")
    query = st.text_area(
        "query",
        placeholder=(
            "e.g.  What's the weather in Mannheim today?\n"
            "      Best restaurants in Tokyo?\n"
            "      Plan a 5-day trip to Portugal for €2000"
        ),
        height=90,
        key="main_query_input",
        label_visibility="collapsed",
    )

    # Detect if it's a planning query to show optional fields
    planning_kw = {"plan", "itinerary", "trip", "days", "week", "schedule", "arrange"}
    is_planning = bool(query.strip()) and any(w in query.lower() for w in planning_kw)

    destination_override = None
    budget_override = None

    if is_planning:
        st.markdown(
            "<p style='font-size:12px;color:#6b7280;margin:2px 0 8px'>🗓️ Planning query detected — optional details below</p>",
            unsafe_allow_html=True,
        )
        dc1, dc2, dc3 = st.columns([2, 2, 2])
        with dc1:
            destination_override = st.text_input(
                "Destination", placeholder="Auto-detected from query",
                key="dest_override", label_visibility="visible"
            ) or None
        with dc2:
            budget_val = st.number_input(
                "Budget (USD)", min_value=0,
                value=st.session_state.user_profile["travel_preferences"]["budget_range"]["max"],
                step=100, key="budget_override"
            )
            budget_override = int(budget_val) if budget_val > 0 else None
        with dc3:
            st.date_input("Start date", value=datetime.now(), key="start_date_override")

    # Submit
    st.markdown("<div style='margin-top:10px'>", unsafe_allow_html=True)
    submitted = st.button(
        "🚀  Get Answer",
        use_container_width=True,
        type="primary",
        key="query_submit",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    if submitted and query.strip():
        if logger is None:
            st.error(f"System not loaded: {import_error}")
            return

        # Reset follow-up on new query
        st.session_state.followup_result = None
        st.session_state.followup_query = None

        with st.spinner("Routing query through agents…"):
            try:
                result = _call_backend(
                    query=query,
                    destination=destination_override,
                    budget=budget_override,
                    user_profile=st.session_state.user_profile,
                )
                st.session_state.last_result = result
                st.session_state.last_query = query

                router = result.get("agent_trace", {}).get("query_router", {})
                st.session_state.query_history.append({
                    "timestamp": datetime.now(),
                    "query": query,
                    "destination": router.get("location") or destination_override or "—",
                    "type": router.get("query_type", "INFORMATION").lower(),
                })
            except Exception as e:
                if logger:
                    logger.error(f"Query failed: {e}", exc_info=True)
                st.error(f"Error: {e}")
                return

    # Show last result (persists across reruns)
    if st.session_state.last_result and st.session_state.last_query:
        _render_result(st.session_state.last_result, st.session_state.last_query)


def render_profile_tab():
    st.markdown("#### Your Travel Profile")

    with st.form("profile_form", border=True):
        c1, c2 = st.columns(2)
        with c1:
            name = st.text_input("Name", value=st.session_state.user_profile.get("name", ""))
            email = st.text_input("Email", value=st.session_state.user_profile.get("email", ""))
        with c2:
            nationality = st.selectbox(
                "Nationality",
                ["US", "UK", "Germany", "France", "India", "Japan", "Canada", "Australia", "Other"],
                index=["US", "UK", "Germany", "France", "India", "Japan",
                       "Canada", "Australia", "Other"].index(
                    st.session_state.user_profile.get("nationality", "US")
                    if st.session_state.user_profile.get("nationality", "US")
                    in ["US", "UK", "Germany", "France", "India", "Japan", "Canada", "Australia"]
                    else "Other"
                ),
            )
            age = st.number_input("Age", min_value=18, max_value=100, value=30)

        st.markdown("**Travel Preferences**")
        bc1, bc2 = st.columns(2)
        with bc1:
            budget_min = st.number_input("Min Budget (USD)", min_value=0, value=1000, step=100)
            trip_duration = st.slider("Preferred trip length (days)", 1, 30, 7)
        with bc2:
            budget_max = st.number_input("Max Budget (USD)", min_value=500, value=5000, step=100)
            travel_pace = st.select_slider("Pace", ["Relaxed", "Moderate", "Fast-paced"], value="Moderate")

        dietary = st.multiselect(
            "Dietary restrictions",
            ["Vegetarian", "Vegan", "Gluten-free", "Nut allergy", "Dairy-free", "Halal", "Kosher"],
        )
        mobility = st.multiselect(
            "Mobility needs",
            ["Wheelchair access", "Limited walking", "Accessible bathrooms"],
        )

        saved = st.form_submit_button("💾  Save Profile", use_container_width=True, type="primary")

    if saved:
        st.session_state.user_profile.update({
            "name": name, "email": email, "nationality": nationality, "age": age,
            "travel_preferences": {
                "budget_range": {"min": budget_min, "max": budget_max},
                "trip_duration": trip_duration,
                "travel_pace": travel_pace,
            },
            "dietary_restrictions": dietary,
            "mobility_needs": mobility,
        })
        st.success("Profile saved!")


def render_history_tab():
    st.markdown("#### Query History")
    history = st.session_state.query_history
    if not history:
        st.info("No queries yet. Ask something in the Query tab!")
        return

    for entry in reversed(history):
        ts = entry["timestamp"].strftime("%d %b %H:%M")
        dest = entry.get("destination", "—")
        qtype = entry.get("type", "").upper()
        with st.expander(f"**{ts}** · {dest} · `{qtype}`"):
            st.markdown(f"_{entry['query']}_")

    if st.button("🗑️  Clear history", type="secondary"):
        st.session_state.query_history = []
        st.rerun()


def _load_history_from_db() -> None:
    """Load query history from SQLite on first page load."""
    if st.session_state.get("_history_loaded"):
        return
    st.session_state._history_loaded = True
    if st.session_state.query_history:
        return  # already populated

    try:
        from src.database import get_db_manager
        from src.database.models import QueryHistory as QH
        with get_db_manager().session_context() as session:
            rows = (
                session.query(QH)
                .order_by(QH.created_at.desc())
                .limit(50)
                .all()
            )
            for row in reversed(rows):
                st.session_state.query_history.append({
                    "timestamp": row.created_at or datetime.now(),
                    "query": row.query_text or "",
                    "destination": (
                        (row.execution_trace or {})
                        .get("query_router", {})
                        .get("location") or "—"
                    ),
                    "type": (row.query_type or "information").lower(),
                })
    except Exception:
        pass  # DB not initialised yet — silent fail


# ── App entry point ───────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="Travel AI",
        page_icon="🌍",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    _init_session()
    _load_history_from_db()

    # ── Global CSS ────────────────────────────────────────────────────────────
    st.markdown(
        """
        <style>
        /* Ensure header is never clipped under the deploy bar */
        .block-container {
            padding-top: 3.5rem !important;
            padding-bottom: 2rem;
            max-width: 1200px;
        }
        /* Hide the default Streamlit top decoration that clips content */
        header[data-testid="stHeader"] {
            background: transparent;
        }

        /* Primary button */
        div[data-testid="stButton"] button[kind="primary"] {
            background: linear-gradient(135deg,#1d4ed8,#2563eb);
            border: none; border-radius: 8px;
            font-weight: 600; font-size: 15px;
            padding: 10px 0; transition: opacity .15s;
        }
        div[data-testid="stButton"] button[kind="primary"]:hover { opacity: .88; }

        /* Secondary / follow-up button — match input height */
        div[data-testid="stButton"] button[kind="secondary"] {
            border-radius: 8px; font-weight: 600;
            height: 38px; padding: 0 16px;
        }

        /* Tabs */
        button[data-baseweb="tab"] {font-weight: 600; font-size: 14px;}

        /* Metric cards */
        div[data-testid="metric-container"] {
            background: #f8fafc; border-radius: 8px;
            padding: 8px 12px; border: 1px solid #e2e8f0;
        }

        /* Text area */
        textarea {border-radius: 8px !important; font-size: 14px !important;}

        /* Expander */
        details {border-radius: 8px !important; border: 1px solid #e2e8f0 !important;}

        /* Follow-up row — align input and button on same baseline */
        div[data-testid="stHorizontalBlock"] div[data-testid="stTextInput"] input {
            height: 38px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # ── Header ────────────────────────────────────────────────────────────────
    name = st.session_state.user_profile.get("name", "Traveller")
    hcol1, hcol2 = st.columns([5, 1])
    with hcol1:
        st.markdown(
            f"""
            <div style="display:flex;align-items:center;gap:12px;padding-bottom:4px">
              <span style="font-size:32px;line-height:1">🌍</span>
              <div>
                <p style="font-size:22px;font-weight:800;color:#0f172a;margin:0;line-height:1.2">
                  Travel Intelligent Agentic System</p>
                <p style="font-size:12px;color:#64748b;margin:0">
                  Multi-agent AI &nbsp;·&nbsp; Real APIs &nbsp;·&nbsp; LLM Knowledge &nbsp;·&nbsp; No hallucination</p>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with hcol2:
        st.markdown(
            f"<div style='text-align:right;padding-top:10px;font-size:13px;color:#64748b'>"
            f"👤 <strong>{name}</strong></div>",
            unsafe_allow_html=True,
        )

    st.markdown("<hr style='margin:8px 0 0'>", unsafe_allow_html=True)

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(["✈️  Query Planner", "👤  Profile", "📚  History"])

    with tab1:
        render_query_tab()
    with tab2:
        render_profile_tab()
    with tab3:
        render_history_tab()

    st.markdown(
        "<hr><p style='text-align:center;font-size:11px;color:#94a3b8'>"
        "Travel Intelligent Agentic System · Multi-Agent Orchestration</p>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()