"""
Comprehensive test suite for Travel Intelligent Agentic System.
Run:  python tests/test_all.py
"""
import sys
import os
import io
import time
import traceback
from datetime import datetime

# Force UTF-8 so the output never crashes on Windows cp1252
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("chromadb").setLevel(logging.ERROR)

# ── Result tracking ──────────────────────────────────────────
PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"
results: list = []


def check(name: str, condition: bool, detail: str = "") -> bool:
    status = PASS if condition else FAIL
    results.append((name, status, str(detail)))
    tag = "[OK]  " if status == PASS else "[FAIL]"
    print(f"  {tag} {name}: {str(detail)[:110]}")
    return condition


def section(title: str) -> None:
    print(f"\n{'='*62}")
    print(f"  {title}")
    print("="*62)


def skip_section(title: str, reason: str) -> None:
    section(title)
    print(f"  [SKIP] {reason}")


# ── Module-level sentinels (never unbound) ───────────────────
settings = None
llm = None
orch = None


# =============================================================
# 1. IMPORTS & SETTINGS
# =============================================================
section("1. IMPORTS & SETTINGS")
try:
    from src.common.settings import get_settings
    settings = get_settings()
    check("Settings loaded", True, f"provider={settings.PRIMARY_LLM_PROVIDER}")
    check("OpenAI key present", bool(settings.OPENAI_API_KEY), "configured")
    check("Weather key present", bool(settings.WEATHER_API_KEY),
          (settings.WEATHER_API_KEY[:8] + "...") if settings.WEATHER_API_KEY else "MISSING")
    check("Duffel key present", bool(settings.BOOKING_API_KEY),
          (settings.BOOKING_API_KEY[:16] + "...") if settings.BOOKING_API_KEY else "MISSING")
except Exception as exc:
    check("Settings loaded", False, str(exc))


# =============================================================
# 2. TOOL HELPERS
# =============================================================
section("2. TOOL HELPERS  (IATA lookup + OWM city query builder)")
try:
    from src.tools.tools import _city_to_iata, _owm_query

    # IATA lookup
    iata_pairs = [
        ("Frankfurt", "FRA"), ("Delhi", "DEL"), ("London", "LHR"),
        ("Paris", "CDG"), ("Tokyo", "NRT"), ("New York", "JFK"),
        ("mumbai", "BOM"), ("frankfurt, germany", "FRA"), ("singapore", "SIN"),
    ]
    for city, expected in iata_pairs:
        check(f"IATA {city!r}", _city_to_iata(city) == expected, f"got {_city_to_iata(city)}")
    check("Unknown city -> None", _city_to_iata("Randomville") is None)

    # OWM query builder — must produce city,ISO2 to prevent wrong-country resolution
    owm_pairs = [
        ("Heidelberg, Germany",  "Heidelberg,DE"),   # was resolving to ZA — critical fix
        ("Heidelberg",           "Heidelberg,DE"),   # ambiguous city in AMBIGUOUS_CITIES
        ("Mannheim",             "Mannheim,DE"),
        ("Paris, France",        "Paris,FR"),
        ("Paris",                "Paris"),            # globally unique — no suffix needed
        ("Tokyo",                "Tokyo"),
        ("New York, US",         "New York,US"),
        ("Berlin",               "Berlin,DE"),
        ("Frankfurt, Germany",   "Frankfurt,DE"),
    ]
    for city, expected in owm_pairs:
        got = _owm_query(city)
        check(f"OWM query {city!r}", got == expected, f"got {got!r}")
except Exception as exc:
    check("Tool helpers", False, traceback.format_exc()[:200])


# =============================================================
# 3. WEATHER TOOL
# =============================================================
section("3. WEATHER TOOL  (OpenWeatherMap -> Open-Meteo fallback)")
try:
    from src.tools.tools import WeatherTool
    wt = WeatherTool()

    # 3a. Major city via OWM
    r = wt.run(destination="Tokyo")
    src = r.get("source", "")
    temp = r.get("current", {}).get("temperature")
    check("Tokyo: API source (not llm_knowledge)", "llm_knowledge" not in src, src)
    check("Tokyo: current temperature present", temp is not None, f"{temp}C")
    check("Tokyo: forecast has >=5 days", len(r.get("forecast", [])) >= 5,
          f"{len(r.get('forecast',[]))} days")

    # 3b. Heidelberg — was resolving to Heidelberg, South Africa (ZA, 18C) — critical fix
    r2 = wt.run(destination="Heidelberg, Germany")
    src2 = r2.get("source", "")
    temp2 = r2.get("current", {}).get("temperature")
    dest2 = r2.get("destination", "")
    check("Heidelberg: resolved to DE not ZA", "ZA" not in dest2 and "DE" in dest2,
          f"resolved={dest2!r}")
    check("Heidelberg: temp in German range 0-30C",
          temp2 is not None and 0 <= float(temp2) <= 30, f"{temp2}C")
    check("Heidelberg: source is live API", "llm_knowledge" not in src2, src2)

    # 3c. Smaller city (OWM may fall through to Open-Meteo)
    r3 = wt.run(destination="Mannheim")
    check("Mannheim: fetched (not llm_knowledge)", "llm_knowledge" not in r3.get("source",""),
          r3.get("source","?"))
    check("Mannheim: temp present",
          r3.get("current", {}).get("temperature") is not None,
          str(r3.get("current",{}).get("temperature")))

    # 3d. Plausible temperature range
    r4 = wt.run(destination="Berlin")
    temp4 = r4.get("current", {}).get("temperature")
    check("Berlin: temp in range -30..50",
          temp4 is not None and -30 < float(temp4) < 50, f"{temp4}C")

    # 3e. Forecast dates >= today
    fc = r4.get("forecast", [])
    today_str = datetime.now().strftime("%Y-%m-%d")
    if fc:
        check("Berlin: forecast dates are not in the past",
              all(str(d.get("date",""))[:10] >= today_str for d in fc),
              f"first={fc[0].get('date','?')}")
        check("Berlin: forecast temps are numbers",
              all(isinstance(d.get("temp_high"), (int, float)) for d in fc),
              "all numeric")

    # 3f. LLM fallback for genuinely unreachable city
    r5 = wt.run(destination="Atlantis Fictional City 99999")
    check("Unreachable city: has source field", "source" in r5, r5.get("source","?"))

except Exception as exc:
    check("WeatherTool", False, traceback.format_exc()[:300])


# =============================================================
# 4. FLIGHTS TOOL
# =============================================================
section("4. FLIGHTS TOOL  (Duffel with IATA codes -> LLM fallback)")
try:
    from src.tools.tools import FlightsTool
    ft = FlightsTool()

    # 4a. The exact case that was failing: city names -> IATA resolved
    r = ft.run(origin="Frankfurt", destination="Delhi", departure_date="2026-07-01")
    src = r.get("source", "")
    check("Frankfurt->Delhi: has source", bool(src), src)
    if "Duffel" in src:
        flights = r.get("flights", [])
        check("Frankfurt->Delhi: Duffel returned flights", len(flights) > 0,
              f"{len(flights)} offers")
        check("Frankfurt->Delhi: IATA origin=FRA", r.get("origin_iata") == "FRA",
              str(r.get("origin_iata")))
        check("Frankfurt->Delhi: IATA dest=DEL", r.get("destination_iata") == "DEL",
              str(r.get("destination_iata")))
        prices = [f.get("price_usd", 0) for f in flights]
        check("Frankfurt->Delhi: prices > 0", all(p > 0 for p in prices), str(prices[:3]))
        check("Frankfurt->Delhi: no fake random IDs",
              all("FL" not in f.get("airline","") for f in flights), "real airlines")
    else:
        check("Frankfurt->Delhi: LLM note present (Duffel failed)", bool(r.get("note","")),
              r.get("note","")[:80])

    # 4b. Another route
    r2 = ft.run(origin="London", destination="New York", departure_date="2026-08-01")
    check("London->NYC: has response", "source" in r2, r2.get("source","?"))

    # 4c. City NOT in IATA map -> must go straight to LLM (no guessed IATA)
    r3 = ft.run(origin="Heidelberg", destination="Mannheim", departure_date="2026-06-01")
    check("Unknown cities: no crash, has source", "source" in r3, r3.get("source","?"))
    check("Unknown cities: source=llm_knowledge (no guessed IATA)",
          "llm_knowledge" in r3.get("source",""), r3.get("source","?"))
    check("Unknown cities: has note for LLM", bool(r3.get("note","")),
          r3.get("note","")[:60])

except Exception as exc:
    check("FlightsTool", False, traceback.format_exc()[:300])


# =============================================================
# 5. RESTAURANTS TOOL
# =============================================================
section("5. RESTAURANTS TOOL  (Overpass/OSM -> LLM fallback)")
try:
    from src.tools.tools import RestaurantsTool
    rtt = RestaurantsTool()

    r = rtt.run(destination="Berlin")
    src = r.get("source", "")
    check("Berlin restaurants: has source", bool(src), src)
    if "OpenStreetMap" in src:
        rlist = r.get("restaurants", [])
        check("Berlin: OSM returned results", len(rlist) > 0, f"{len(rlist)} places")
        check("Berlin: each has a name", all(bool(x.get("name")) for x in rlist), "names ok")
    else:
        check("Berlin: LLM note present", bool(r.get("note","")), src)

    # Smaller city -> LLM note
    r2 = rtt.run(destination="Heidelberg")
    check("Heidelberg: has source", "source" in r2, r2.get("source","?"))

    # With cuisine filter
    r3 = rtt.run(destination="Tokyo", cuisine="sushi")
    check("Tokyo sushi: has source", "source" in r3, r3.get("source","?"))

except Exception as exc:
    check("RestaurantsTool", False, traceback.format_exc()[:200])


# =============================================================
# 6. HOTELS TOOL
# =============================================================
section("6. HOTELS TOOL  (OSM live -> LLM knowledge fallback)")
try:
    from src.tools.tools import HotelsTool
    ht = HotelsTool()
    r = ht.run(destination="Barcelona")
    src = r.get("source", "")
    check("Barcelona hotels: has source", bool(src), src)
    check("Barcelona hotels: has note", bool(r.get("note","")), r.get("note","")[:80])
    check("Barcelona hotels: destination in result", r.get("destination","").lower() == "barcelona")
    # If OSM returned real hotels, verify structure
    if "OpenStreetMap" in src:
        hotels = r.get("hotels", [])
        check("Barcelona hotels: OSM returned hotels", len(hotels) > 0, f"{len(hotels)} hotels")
        check("Barcelona hotels: each has name", all(bool(h.get("name")) for h in hotels), "names ok")
    else:
        check("Barcelona hotels: LLM fallback", "llm_knowledge" in src, src)
except Exception as exc:
    check("HotelsTool", False, traceback.format_exc()[:200])


# =============================================================
# 7. VISA TOOL
# =============================================================
section("7. VISA TOOL  (REST Countries API + LLM note)")
try:
    from src.tools.tools import VisaTool
    vt = VisaTool()

    r = vt.run(origin_country="Germany", destination_country="Japan")
    check("Germany->Japan: has LLM note", bool(r.get("note","")), r.get("note","")[:80])
    ci = r.get("country_info", {})
    check("Japan country_info fetched", isinstance(ci, dict), str(ci))
    check("Japan capital = Tokyo", "tokyo" in ci.get("capital","").lower(),
          ci.get("capital","?"))
    check("Japan currency = JPY", "JPY" in ci.get("currency",""),
          ci.get("currency","?"))

    # Unknown country -> graceful
    r2 = vt.run(origin_country="US", destination_country="Fakeland99")
    check("Unknown country: no crash", "source" in r2, r2.get("source","?"))
    check("Unknown country: has note", bool(r2.get("note","")), "ok")

except Exception as exc:
    check("VisaTool", False, traceback.format_exc()[:200])


# =============================================================
# 8. HEALTH TOOL
# =============================================================
section("8. HEALTH TOOL  (LLM knowledge)")
try:
    from src.tools.tools import HealthTool
    het = HealthTool()
    r = het.run(destination_country="India")
    check("India health: source=llm_knowledge", "llm_knowledge" in r.get("source",""),
          r.get("source","?"))
    check("India health: note instructs LLM", bool(r.get("note","")),
          r.get("note","")[:80])
    check("India health: destination present", r.get("destination","").lower() == "india")
except Exception as exc:
    check("HealthTool", False, traceback.format_exc()[:200])


# =============================================================
# 9. TRANSPORT TOOL
# =============================================================
section("9. TRANSPORT TOOL  (LLM knowledge)")
try:
    from src.tools.tools import TransportTool
    trt = TransportTool()
    r = trt.run(destination="Paris")
    check("Paris transport: source=llm_knowledge", "llm_knowledge" in r.get("source",""),
          r.get("source","?"))
    check("Paris transport: note instructs LLM", bool(r.get("note","")),
          r.get("note","")[:80])
except Exception as exc:
    check("TransportTool", False, traceback.format_exc()[:200])


# =============================================================
# 10. CULTURAL INFO TOOL
# =============================================================
section("10. CULTURAL INFO TOOL  (REST Countries + LLM)")
try:
    from src.tools.tools import CulturalInfoTool
    cit = CulturalInfoTool()
    r = cit.run(destination_country="Japan")
    check("Japan culture: has note", bool(r.get("note","")), r.get("note","")[:80])
    cf = r.get("country_facts", {})
    check("Japan: country_facts from REST", isinstance(cf, dict) and bool(cf),
          str(cf.get("capital","?")))
    check("Japan: capital = Tokyo", "tokyo" in str(cf.get("capital","")).lower(),
          cf.get("capital","?"))
except Exception as exc:
    check("CulturalInfoTool", False, traceback.format_exc()[:200])


# =============================================================
# 11. QUERY ROUTER
# =============================================================
section("11. QUERY ROUTER  (LLM classification)")

if settings is None:
    skip_section("11. QUERY ROUTER", "settings unavailable")
else:
    try:
        from src.llm import LLMFactory
        from src.agents.agents import QueryRouter

        llm = LLMFactory.create_from_settings(settings)
        qr = QueryRouter(llm)

        router_cases = [
            # (query, exp_type_options, exp_tool, exp_loc_options)
            # exp_loc_options: list of acceptable location strings (city OR country)
            ("What is the weather in Tokyo?",
             ["INFORMATION"],            "weather",       ["Tokyo"]),
            ("Give me flights from Frankfurt to Delhi",
             ["INFORMATION"],            "flights",       ["Delhi"]),
            ("What restaurants are in Berlin?",
             ["INFORMATION","RECOMMENDATION"], "restaurants", ["Berlin"]),
            ("Visa requirements for India from Germany",
             ["INFORMATION"],            "visa",          ["India", "Delhi", "Mumbai"]),
            ("Plan a 7-day trip to Japan",
             ["PLANNING"],               None,            ["Japan", "Tokyo"]),
            ("Recommend things to do in Singapore",
             ["RECOMMENDATION"],         None,            ["Singapore"]),
            ("Is it safe health-wise to visit Thailand",
             ["INFORMATION","RECOMMENDATION"], "health",  ["Thailand", "Bangkok"]),
            ("How to get around Paris by metro",
             ["INFORMATION"],            "transport",     ["Paris"]),
            ("Cultural customs in India",
             ["INFORMATION"],            "cultural_info", ["India", "Delhi", "Mumbai", "New Delhi"]),
            ("What hotels are available in Barcelona?",
             ["INFORMATION","RECOMMENDATION"], "hotels",  ["Barcelona"]),
        ]

        for query, exp_types, exp_tool, exp_locs in router_cases:
            try:
                r = qr.execute(query=query)
                got_type  = r.get("query_type", "")
                got_loc   = (r.get("location") or r.get("destination_city") or "").lower()
                got_country = (r.get("location_country") or "").lower()
                got_tools = r.get("tools_needed", [])
                type_ok   = got_type in exp_types
                # Location can be city OR country
                loc_ok    = any(
                    exp.lower() in got_loc or exp.lower() in got_country
                    for exp in exp_locs
                )
                tool_ok   = exp_tool is None or exp_tool in got_tools
                check(
                    f"Route: {query[:48]}",
                    type_ok and loc_ok and tool_ok,
                    f"type={got_type} loc={got_loc!r} country={got_country!r} tools={got_tools}",
                )
            except Exception as exc:
                check(f"Route: {query[:48]}", False, str(exc)[:100])

    except Exception as exc:
        check("QueryRouter init", False, traceback.format_exc()[:200])


# =============================================================
# 12. END-TO-END FLOWS
# =============================================================
section("12. END-TO-END FLOWS  (full orchestrator)")

if llm is None:
    skip_section("12. END-TO-END FLOWS", "LLM unavailable - skipping")
else:
    try:
        from src.agents.orchestrator import AgentOrchestrator
        from src.tools.tools import get_all_tools

        tools = get_all_tools()
        orch = AgentOrchestrator(llm=llm, tools=tools, enable_rag=False)

        e2e_cases = [
            # (query, acceptable_agent_keys, keywords_in_answer)
            ("What is the weather in Berlin?",
             ["information_agent"], ["berlin"]),
            ("Give me flights from London to Mumbai",
             ["information_agent"], ["london", "mumbai"]),
            # hotels/restaurants may go to info OR recommendation agent
            ("What are hotel options in Barcelona?",
             ["information_agent","recommendation_agent"], ["barcelona"]),
            ("Visa requirements to visit Japan from the US",
             ["information_agent"], ["japan"]),
            ("Is Thailand safe to visit health-wise?",
             ["information_agent"], ["thailand", "bangkok", "health", "safe"]),
            ("How do I get around in Tokyo?",
             ["information_agent"], ["tokyo"]),
            # LLM may answer "Germany" or "Berlin" for cultural customs
            ("What are cultural customs in Germany?",
             ["information_agent"], ["german", "germany", "berlin"]),
            ("Plan a 3-day trip to Paris",
             ["planning_agent"], ["paris", "day"]),
            ("Recommend things to do in Singapore",
             ["recommendation_agent"], ["singapore"]),
            ("Flights from Frankfurt to New York",
             ["information_agent"], ["frankfurt", "new york"]),
        ]

        for query, exp_agents, must_have in e2e_cases:
            try:
                t0 = time.time()
                result = orch.execute(query=query)
                elapsed = round(time.time() - t0, 1)

                trace = result.get("agent_trace", {})
                agent_ran = any(a in trace for a in exp_agents)
                answer = (result.get("response") or "").lower()
                has_answer = bool(answer) and "unable to process" not in answer
                # must_have: all must appear OR any one of the alternatives appears
                content_ok = any(kw.lower() in answer for kw in must_have)
                conf = result.get("validation", {}).get("confidence_score", 0)

                check(
                    f"E2E: {query[:50]}",
                    agent_ran and has_answer and content_ok,
                    f"agent={'ok' if agent_ran else 'MISSING'} "
                    f"content={'ok' if content_ok else 'MISSING'} "
                    f"conf={conf} {elapsed}s",
                )
            except Exception as exc:
                check(f"E2E: {query[:50]}", False, str(exc)[:120])

    except Exception as exc:
        check("Orchestrator init", False, traceback.format_exc()[:300])


# =============================================================
# 13. EDGE CASES
# =============================================================
section("13. EDGE CASES")

if orch is None:
    skip_section("13. EDGE CASES", "orchestrator unavailable - skipping")
else:
    try:
        # 13a. Non-travel query
        r = orch.execute(query="What is 2 + 2?")
        answer = (r.get("response") or "").lower()
        check("Non-travel query: handled, not empty", bool(answer), answer[:80])

        # 13b. Missing destination
        r2 = orch.execute(query="What is the weather today?")
        check("No destination: no crash", bool(r2.get("response", "")))

        # 13c. Origin-only flight query
        r3 = orch.execute(query="Flights from Berlin")
        check("Origin-only flight: no crash", bool(r3.get("response", "")))

        # 13d. Planning query invokes planning_agent
        r4 = orch.execute(query="Plan a 5-day trip to Thailand with a budget of 2000 USD")
        trace4 = r4.get("agent_trace", {})
        check("Planning: planning_agent invoked", "planning_agent" in trace4)
        check("Planning: response length > 200 chars",
              len(r4.get("response", "")) > 200,
              f"{len(r4.get('response',''))} chars")

        # 13e. Typo in city name
        r5 = orch.execute(query="Weather in Frankfurtt Germany")
        check("Typo city: no crash, has response", bool(r5.get("response", "")))

        # 13f. Tool cache - second call should be faster (tool result cached)
        t1 = time.time()
        orch.execute(query="What is the weather in Amsterdam?")
        first_s = round(time.time() - t1, 1)

        t2 = time.time()
        orch.execute(query="What is the weather in Amsterdam?")
        second_s = round(time.time() - t2, 1)
        # Tool cache should make the second call meaningfully faster
        check("Tool cache: second call faster or comparable",
              second_s <= first_s + 2,   # generous tolerance - LLM still runs
              f"1st={first_s}s 2nd={second_s}s")

        # 13g. Recommendation agent invoked
        r6 = orch.execute(query="Recommend what to do in Kyoto, Japan")
        trace6 = r6.get("agent_trace", {})
        check("Recommendation: recommendation_agent invoked",
              "recommendation_agent" in trace6,
              str(list(trace6.keys())))

    except Exception as exc:
        check("Edge cases", False, traceback.format_exc()[:300])


# =============================================================
# 14. DATA QUALITY ASSERTIONS
# =============================================================
section("14. DATA QUALITY")
try:
    from src.tools.tools import WeatherTool, FlightsTool, VisaTool
    wq = WeatherTool()
    fq = FlightsTool()
    vq = VisaTool()

    # 14a. Temperature sanity
    rw = wq.run(destination="Berlin")
    temp = rw.get("current", {}).get("temperature")
    check("Berlin: temp is numeric", isinstance(temp, (int, float)), str(temp))
    check("Berlin: temp in -30..50 range",
          temp is not None and -30 < float(temp) < 50, f"{temp}C")

    # 14b. Forecast structure
    fc = rw.get("forecast", [])
    if fc:
        today_str = datetime.now().strftime("%Y-%m-%d")
        check("Forecast: dates not in past",
              all(str(d.get("date", ""))[:10] >= today_str for d in fc),
              f"first={fc[0].get('date','?')}")
        check("Forecast: temp_high/temp_low are numbers",
              all(isinstance(d.get("temp_high"), (int, float)) and
                  isinstance(d.get("temp_low"),  (int, float)) for d in fc),
              "all numeric")
        check("Forecast: high >= low for all days",
              all(d["temp_high"] >= d["temp_low"] for d in fc), "temperatures consistent")

    # 14c. Flights data quality
    rf = fq.run(origin="Frankfurt", destination="Delhi", departure_date="2026-08-01")
    if "Duffel" in rf.get("source", ""):
        flights = rf.get("flights", [])
        check("Duffel: at least 1 flight returned", len(flights) > 0, f"{len(flights)} offers")
        check("Duffel: prices > 0",
              all(f.get("price_usd", 0) > 0 for f in flights),
              str([round(f.get("price_usd",0),2) for f in flights[:3]]))
        check("Duffel: no fake 'FL' airline IDs",
              all("FL" not in str(f.get("airline","")) for f in flights), "real airline names")
    else:
        check("Flights: LLM note present (Duffel fallback)", bool(rf.get("note","")), "ok")

    # 14d. Visa REST Countries real data
    rv = vq.run(origin_country="US", destination_country="France")
    ci = rv.get("country_info", {})
    check("France: country_info returned", isinstance(ci, dict) and bool(ci), str(ci))
    check("France: capital = Paris",
          "paris" in str(ci.get("capital", "")).lower(), ci.get("capital", "?"))
    check("France: currency = EUR",
          "EUR" in str(ci.get("currency", "")), ci.get("currency", "?"))

    # 14e. No random data in health / transport
    from src.tools.tools import HealthTool, TransportTool
    rh = HealthTool().run(destination_country="Thailand")
    rt = TransportTool().run(destination="Bangkok")
    check("Health: no random data (source=llm_knowledge)",
          "llm_knowledge" in rh.get("source", ""), rh.get("source","?"))
    check("Transport: no random data (source=llm_knowledge)",
          "llm_knowledge" in rt.get("source", ""), rt.get("source","?"))

except Exception as exc:
    check("Data quality", False, traceback.format_exc()[:300])


# =============================================================
# SUMMARY
# =============================================================
section("SUMMARY")
passed  = sum(1 for _, s, _ in results if s == PASS)
failed  = sum(1 for _, s, _ in results if s == FAIL)
total   = len(results)

print(f"\n  Total  : {total}")
print(f"  Passed : {passed}")
print(f"  Failed : {failed}")

if failed:
    print("\n  FAILED TESTS:")
    for name, status, detail in results:
        if status == FAIL:
            print(f"    [FAIL] {name}")
            if detail:
                print(f"           {detail[:200]}")

print()
sys.exit(0 if failed == 0 else 1)