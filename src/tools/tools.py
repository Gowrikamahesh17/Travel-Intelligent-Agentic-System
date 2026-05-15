"""
Travel tool implementations.
Each tool attempts real API calls first, then signals the LLM to answer from knowledge.
No random or fabricated data is ever generated here.
"""

from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import requests
from .base import BaseTool
from src.common import get_settings

# City name → IATA airport code for common cities
CITY_TO_IATA = {
    "frankfurt": "FRA", "berlin": "BER", "munich": "MUC", "hamburg": "HAM",
    "cologne": "CGN", "dusseldorf": "DUS", "stuttgart": "STR",
    "london": "LHR", "paris": "CDG", "amsterdam": "AMS", "rome": "FCO",
    "madrid": "MAD", "barcelona": "BCN", "lisbon": "LIS", "zurich": "ZRH",
    "vienna": "VIE", "brussels": "BRU", "oslo": "OSL", "stockholm": "ARN",
    "copenhagen": "CPH", "helsinki": "HEL", "warsaw": "WAW", "prague": "PRG",
    "budapest": "BUD", "athens": "ATH", "istanbul": "IST",
    "new york": "JFK", "los angeles": "LAX", "chicago": "ORD",
    "san francisco": "SFO", "miami": "MIA", "boston": "BOS",
    "washington": "IAD", "seattle": "SEA", "dallas": "DFW",
    "toronto": "YYZ", "montreal": "YUL", "vancouver": "YVR",
    "delhi": "DEL", "new delhi": "DEL", "mumbai": "BOM", "bangalore": "BLR",
    "hyderabad": "HYD", "chennai": "MAA", "kolkata": "CCU",
    "dubai": "DXB", "abu dhabi": "AUH", "doha": "DOH", "riyadh": "RUH",
    "tokyo": "NRT", "osaka": "KIX", "beijing": "PEK", "shanghai": "PVG",
    "hong kong": "HKG", "singapore": "SIN", "bangkok": "BKK",
    "kuala lumpur": "KUL", "jakarta": "CGK", "manila": "MNL",
    "sydney": "SYD", "melbourne": "MEL", "auckland": "AKL",
    "cairo": "CAI", "johannesburg": "JNB", "nairobi": "NBO",
    "casablanca": "CMN", "lagos": "LOS",
    "reykjavik": "KEF", "oslo": "OSL", "helsinki": "HEL",
    "athens": "ATH", "lisbon": "LIS", "zurich": "ZRH",
    "mexico city": "MEX", "sao paulo": "GRU", "buenos aires": "EZE",
    "bogota": "BOG", "lima": "LIM", "santiago": "SCL",
}


def _extract_city(text: str) -> str:
    """
    Extract just the city name from strings like:
      "Frankfurt Germany" → "Frankfurt"
      "Delhi India"       → "Delhi"
      "Frankfurt, Germany"→ "Frankfurt"
      "New York"          → "New York"      (multi-word city, no country)
      "New York, US"      → "New York"

    Strategy: split on comma first, then check if the last word is a known
    country name and strip it.
    """
    # Strip comma-separated country suffix first
    parts = [p.strip() for p in text.split(",")]
    city_part = parts[0].strip()

    # Now check if the last word(s) of city_part are a known country name
    words = city_part.split()
    if len(words) >= 2:
        # Try progressively longer country suffixes (handles "United Kingdom", "South Africa")
        for n in range(len(words) - 1, 0, -1):
            suffix = " ".join(words[n:]).lower()
            if suffix in COUNTRY_NAME_TO_ISO2 or suffix in {
                "india", "germany", "france", "japan", "china", "usa",
                "uk", "italy", "spain", "australia", "canada", "brazil",
                "russia", "mexico", "thailand", "indonesia", "turkey",
                "saudi arabia", "egypt", "south africa", "nigeria",
                "argentina", "colombia", "korea",
            }:
                city_part = " ".join(words[:n])
                break

    return city_part.strip()


def _city_to_iata(city: str) -> Optional[str]:
    """Return IATA code for a city name. Handles 'Frankfurt Germany' → FRA."""
    city_clean = _extract_city(city).lower().strip()
    return CITY_TO_IATA.get(city_clean)


# Country full name → ISO 3166-1 alpha-2 code
# OWM requires the 2-letter code for unambiguous city resolution
COUNTRY_NAME_TO_ISO2 = {
    "germany": "DE", "france": "FR", "united kingdom": "GB", "uk": "GB",
    "spain": "ES", "italy": "IT", "portugal": "PT", "netherlands": "NL",
    "belgium": "BE", "switzerland": "CH", "austria": "AT", "sweden": "SE",
    "norway": "NO", "denmark": "DK", "finland": "FI", "poland": "PL",
    "czech republic": "CZ", "czechia": "CZ", "hungary": "HU", "greece": "GR",
    "romania": "RO", "bulgaria": "BG", "croatia": "HR", "slovakia": "SK",
    "slovenia": "SI", "serbia": "RS", "ukraine": "UA", "russia": "RU",
    "turkey": "TR", "ireland": "IE", "luxembourg": "LU", "iceland": "IS",
    "united states": "US", "usa": "US", "us": "US", "canada": "CA",
    "mexico": "MX", "brazil": "BR", "argentina": "AR", "chile": "CL",
    "colombia": "CO", "peru": "PE", "venezuela": "VE",
    "india": "IN", "china": "CN", "japan": "JP", "south korea": "KR",
    "indonesia": "ID", "thailand": "TH", "vietnam": "VN", "malaysia": "MY",
    "singapore": "SG", "philippines": "PH", "bangladesh": "BD", "pakistan": "PK",
    "australia": "AU", "new zealand": "NZ",
    "south africa": "ZA", "nigeria": "NG", "kenya": "KE", "egypt": "EG",
    "morocco": "MA", "ethiopia": "ET", "ghana": "GH", "tanzania": "TZ",
    "saudi arabia": "SA", "uae": "AE", "united arab emirates": "AE",
    "qatar": "QA", "kuwait": "KW", "israel": "IL", "iran": "IR", "iraq": "IQ",
}


def _owm_query(city: str) -> str:
    """
    Convert a city string (possibly 'City, Country') to OWM's preferred 'City,ISO2' format.
    OWM uses ISO-2 country codes for unambiguous lookup — full country names cause wrong matches.

    Examples:
      'Heidelberg, Germany'  -> 'Heidelberg,DE'
      'Paris, France'        -> 'Paris,FR'
      'New York, US'         -> 'New York,US'   (already ISO-2)
      'Berlin'               -> 'Berlin,DE'  (known German city → append DE)
      'Tokyo'                -> 'Tokyo'      (globally unique, no suffix needed)
    """
    # Cities that share a name with cities in other countries — must have country code
    AMBIGUOUS_CITIES = {
        "heidelberg": "DE", "mannheim": "DE", "frankfurt": "DE",
        "munich": "DE", "cologne": "DE", "hamburg": "DE", "berlin": "DE",
        "bremen": "DE", "hanover": "DE", "hannover": "DE", "nuremberg": "DE",
        "stuttgart": "DE", "dortmund": "DE", "essen": "DE", "freiburg": "DE",
        "augsburg": "DE", "wiesbaden": "DE", "bonn": "DE", "dusseldorf": "DE",
        "mannheim": "DE", "karlsruhe": "DE", "worms": "DE", "mainz": "DE",
        "trier": "DE", "koblenz": "DE", "erfurt": "DE", "jena": "DE",
        "richmond": "GB", "springfield": "US", "cambridge": "GB",
        "birmingham": "GB", "newcastle": "GB", "bath": "GB",
        "reading": "GB", "oxford": "GB",
        "florence": "IT", "milan": "IT", "naples": "IT", "venice": "IT",
        "leon": "ES", "granada": "ES", "victoria": "AU",
    }

    parts = [p.strip() for p in city.split(",")]
    city_name = parts[0]
    city_lower = city_name.lower()

    # If a country suffix was provided, convert it to ISO-2
    if len(parts) >= 2:
        suffix = parts[-1].strip()
        # Already a 2-letter ISO code?
        if len(suffix) == 2 and suffix.isalpha():
            return f"{city_name},{suffix.upper()}"
        # Convert full name to ISO-2
        iso2 = COUNTRY_NAME_TO_ISO2.get(suffix.lower())
        if iso2:
            return f"{city_name},{iso2}"
        # Unknown suffix — drop it (OWM may still resolve correctly)
        return city_name

    # No country given — check if the city is known to be ambiguous
    iso2 = AMBIGUOUS_CITIES.get(city_lower)
    if iso2:
        return f"{city_name},{iso2}"

    # Globally unique names — pass as-is
    return city_name


class WeatherTool(BaseTool):
    """Fetch current weather and forecast via OpenWeatherMap (paid) or Open-Meteo (free)."""

    def __init__(self):
        super().__init__(name="weather", description="Fetch weather forecast", cache_ttl_seconds=1800)  # 30 min cache
        self.settings = get_settings()
        self.owm_key = self.settings.WEATHER_API_KEY
        self.owm_base = self.settings.WEATHER_API_BASE_URL

    def execute(self, destination: str, days_ahead: int = 7, **kwargs) -> Dict[str, Any]:
        # Strip country name suffix before building OWM query
        # e.g. "Delhi India" → "Delhi", "Frankfurt Germany" → "Frankfurt"
        clean = _extract_city(destination)
        owm_query = _owm_query(clean)
        self.logger.info(f"Weather query: {destination!r} -> cleaned={clean!r} OWM={owm_query!r}")

        # --- Try OpenWeatherMap (paid, configured key) ---
        if self.owm_key:
            result = self._fetch_owm(owm_query, destination, days_ahead)
            if result:
                return result

        # --- Try Open-Meteo + geocoding (completely free, no key) ---
        result = self._fetch_open_meteo(clean, days_ahead)
        if result:
            return result

        # --- Signal LLM to use its own knowledge ---
        self.logger.warning(f"All weather APIs failed for {destination}, signalling LLM fallback")
        return {
            "destination": destination,
            "source": "llm_knowledge",
            "note": "Live weather unavailable. Answer from training knowledge with a disclaimer."
        }

    def _fetch_owm(self, owm_query: str, original_dest: str, days_ahead: int) -> Optional[Dict[str, Any]]:
        try:
            params = {"q": owm_query, "appid": self.owm_key, "units": "metric"}
            cur = requests.get(f"{self.owm_base}/weather", params=params, timeout=8).json()
            if "main" not in cur:
                return None

            # Sanity-check: if the query contained a country code, verify OWM
            # resolved to that country (prevents Heidelberg,DE → Heidelberg,ZA)
            if "," in owm_query:
                expected_iso = owm_query.split(",")[-1].upper()
                returned_iso = cur.get("sys", {}).get("country", "").upper()
                if returned_iso and returned_iso != expected_iso:
                    self.logger.warning(
                        f"OWM country mismatch for {owm_query!r}: "
                        f"expected {expected_iso} got {returned_iso} — falling back to Open-Meteo"
                    )
                    return None  # trigger Open-Meteo fallback

            # Fetch 5-day / 3-hour forecast
            fc_resp = requests.get(f"{self.owm_base}/forecast", params=params, timeout=8)
            fc_resp.raise_for_status()
            fc_raw = fc_resp.json()

            daily: Dict[str, Any] = {}
            for entry in fc_raw.get("list", []):
                day = entry["dt_txt"][:10]
                daily.setdefault(day, {"temps": [], "conditions": [], "humidities": []})
                daily[day]["temps"].append(entry["main"]["temp"])
                daily[day]["conditions"].append(entry["weather"][0]["main"])
                daily[day]["humidities"].append(entry["main"]["humidity"])

            forecast = []
            for day, vals in sorted(daily.items())[:days_ahead]:
                forecast.append({
                    "date": day,
                    "temp_high": round(max(vals["temps"]), 1),
                    "temp_low": round(min(vals["temps"]), 1),
                    "condition": max(set(vals["conditions"]), key=vals["conditions"].count),
                    "humidity": round(sum(vals["humidities"]) / len(vals["humidities"])),
                })

            resolved_name = cur.get("name", original_dest)
            resolved_country = cur.get("sys", {}).get("country", "")
            display_name = f"{resolved_name}, {resolved_country}" if resolved_country else resolved_name
            self.logger.info(f"OpenWeatherMap: {original_dest!r} -> {display_name} {cur['main']['temp']}C")

            return {
                "destination": display_name,
                "current": {
                    "temperature": round(cur["main"]["temp"], 1),
                    "feels_like": round(cur["main"]["feels_like"], 1),
                    "humidity": cur["main"]["humidity"],
                    "condition": cur["weather"][0]["main"],
                    "description": cur["weather"][0]["description"],
                    "wind_speed": cur["wind"]["speed"],
                },
                "forecast": forecast,
                "source": "OpenWeatherMap API (live)"
            }
        except Exception as e:
            self.logger.debug(f"OpenWeatherMap failed: {e}")
            return None

    def _fetch_open_meteo(self, destination: str, days_ahead: int) -> Optional[Dict[str, Any]]:
        """Free weather via Open-Meteo + Open-Meteo geocoding (no API key needed)."""
        try:
            # Step 1: geocode
            geo = requests.get(
                "https://geocoding-api.open-meteo.com/v1/search",
                params={"name": destination, "count": 1, "language": "en", "format": "json"},
                timeout=6,
            ).json()
            results = geo.get("results")
            if not results:
                return None
            lat, lon = results[0]["latitude"], results[0]["longitude"]
            resolved_name = results[0].get("name", destination)

            # Step 2: fetch forecast
            fc = requests.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": lat,
                    "longitude": lon,
                    "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max,weathercode",
                    "current_weather": True,
                    "timezone": "auto",
                    "forecast_days": min(days_ahead, 7),
                },
                timeout=8,
            ).json()

            current_w = fc.get("current_weather", {})
            daily_d = fc.get("daily", {})

            wmo_map = {
                0: "Clear", 1: "Mainly clear", 2: "Partly cloudy", 3: "Cloudy",
                45: "Foggy", 48: "Icy fog", 51: "Light drizzle", 61: "Light rain",
                63: "Rain", 65: "Heavy rain", 71: "Light snow", 73: "Snow", 75: "Heavy snow",
                80: "Rain showers", 95: "Thunderstorm",
            }

            forecast = []
            dates = daily_d.get("time", [])
            for i, date in enumerate(dates):
                code = daily_d.get("weathercode", [0])[i] if i < len(daily_d.get("weathercode", [])) else 0
                forecast.append({
                    "date": date,
                    "temp_high": daily_d.get("temperature_2m_max", [None])[i],
                    "temp_low": daily_d.get("temperature_2m_min", [None])[i],
                    "condition": wmo_map.get(code, "Variable"),
                    "precipitation_probability": daily_d.get("precipitation_probability_max", [None])[i],
                })

            self.logger.info(f"Open-Meteo data fetched for {destination} ({resolved_name})")
            return {
                "destination": resolved_name,
                "current": {
                    "temperature": current_w.get("temperature"),
                    "wind_speed": current_w.get("windspeed"),
                    "condition": wmo_map.get(current_w.get("weathercode", 0), "Variable"),
                },
                "forecast": forecast,
                "source": "Open-Meteo (free, live)"
            }
        except Exception as e:
            self.logger.debug(f"Open-Meteo failed: {e}")
            return None


class FlightsTool(BaseTool):
    """Search flights: Duffel API → Aviation Stack → LLM knowledge."""

    def __init__(self):
        super().__init__(name="flights", description="Search available flights", cache_ttl_seconds=3600)
        self.settings = get_settings()
        self.api_key = self.settings.BOOKING_API_KEY

    def execute(self, origin: str, destination: str, departure_date: str,
                return_date: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        # Strip country names before IATA lookup: "Frankfurt Germany" → "Frankfurt"
        origin_city = _extract_city(origin)
        dest_city   = _extract_city(destination)
        origin_iata = _city_to_iata(origin_city)
        dest_iata   = _city_to_iata(dest_city)
        self.logger.info(
            f"Flight search: {origin!r}→{origin_city!r}({origin_iata or 'unknown'}) "
            f"-> {destination!r}→{dest_city!r}({dest_iata or 'unknown'})"
        )

        # 1. Duffel (requires both IATA codes — never guess)
        if self.api_key and origin_iata and dest_iata:
            result = self._fetch_duffel(origin_iata, dest_iata, origin_city, dest_city, departure_date)
            if result:
                return result

        # 2. Signal LLM — it knows airlines/routes well
        self.logger.info(f"Signalling LLM for flight knowledge: {origin_city} -> {dest_city}")
        origin_label = f"{origin_city} ({origin_iata})" if origin_iata else origin_city
        dest_label = f"{dest_city} ({dest_iata})" if dest_iata else dest_city
        return {
            "origin": origin_city, "origin_iata": origin_iata,
            "destination": dest_city, "destination_iata": dest_iata,
            "departure_date": departure_date,
            "source": "llm_knowledge",
            "note": (
                f"Provide flight information from {origin_label} to {dest_label} "
                f"from your training knowledge. "
                f"Include: which airlines operate this route, typical flight duration, "
                f"number of stops, approximate price range in EUR/USD, and booking tips. "
                f"Disclaimer: 'Prices and schedules change — check Google Flights, Skyscanner, "
                f"or airline websites for current fares.'"
            )
        }

    def _fetch_duffel(self, origin_iata: str, dest_iata: str,
                      origin_city: str, dest_city: str, departure_date: str) -> Optional[Dict[str, Any]]:
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Duffel-Version": "v2",
                "Content-Type": "application/json",
                "Accept": "application/json",
            }
            payload = {
                "data": {
                    "slices": [{"origin": origin_iata, "destination": dest_iata,
                                "departure_date": departure_date}],
                    "passengers": [{"type": "adult"}],
                    "cabin_class": "economy",
                }
            }
            resp = requests.post("https://api.duffel.com/air/offer_requests",
                                 json=payload, headers=headers, timeout=15)
            if resp.status_code not in (200, 201):
                self.logger.debug(f"Duffel {resp.status_code}: {resp.text[:200]}")
                return None

            offer_request_id = resp.json()["data"]["id"]
            offers_resp = requests.get(
                f"https://api.duffel.com/air/offers?offer_request_id={offer_request_id}&limit=5",
                headers=headers, timeout=15)
            offers = offers_resp.json().get("data", [])
            if not offers:
                return None

            flights = []
            for offer in offers[:5]:
                slice_ = offer["slices"][0]
                seg = slice_["segments"][0]
                flights.append({
                    "airline": seg.get("operating_carrier", {}).get("name", "Unknown"),
                    "flight_number": seg.get("operating_carrier_flight_number", ""),
                    "departure_time": seg.get("departing_at", "")[:16],
                    "arrival_time": seg.get("arriving_at", "")[:16],
                    "duration": slice_.get("duration", ""),
                    "stops": len(slice_["segments"]) - 1,
                    "price_usd": float(offer.get("total_amount", 0)),
                    "currency": offer.get("total_currency", "USD"),
                })

            self.logger.info(f"Duffel: {len(flights)} flights {origin_iata}→{dest_iata}")
            return {
                "origin": origin_city, "origin_iata": origin_iata,
                "destination": dest_city, "destination_iata": dest_iata,
                "departure_date": departure_date, "flights": flights,
                "source": "Duffel API (live)",
            }
        except Exception as e:
            self.logger.debug(f"Duffel error: {e}")
            return None



def _geocode_nominatim(place: str) -> Optional[Dict]:
    """Geocode a city name via Nominatim (OpenStreetMap, free, no key)."""
    try:
        geo = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": place, "format": "json", "limit": 1},
            headers={"User-Agent": "TravelAI/1.0"},
            timeout=6,
        ).json()
        if geo:
            return {"lat": float(geo[0]["lat"]), "lon": float(geo[0]["lon"]),
                    "display_name": geo[0].get("display_name", place)}
    except Exception:
        pass
    return None


def _overpass_query(query: str, timeout: int = 15) -> list:
    """Execute an Overpass QL query using GET (POST returns 406). Free, no key."""
    resp = requests.get(
        "https://overpass-api.de/api/interpreter",
        params={"data": query},
        headers={"User-Agent": "TravelAI/1.0"},
        timeout=timeout,
    )
    if resp.status_code != 200:
        return []
    return resp.json().get("elements", [])


class HotelsTool(BaseTool):
    """
    Hotel search:
      1. Overpass/OSM — returns real accommodation POIs (hotels, hostels, guesthouses)
      2. LLM knowledge — if OSM returns too few results
    """

    def __init__(self):
        super().__init__(name="hotels", description="Find hotel accommodations",
                         cache_ttl_seconds=3600)
        self.settings = get_settings()

    def execute(self, destination: str, check_in: Optional[str] = None,
                check_out: Optional[str] = None, guests: int = 1, **kwargs) -> Dict[str, Any]:
        destination = _extract_city(destination)  # strip "Delhi India" → "Delhi"
        # Try OpenStreetMap Overpass for real accommodation data (free, no key)
        result = self._fetch_osm_hotels(destination)
        if result:
            return result

        # Signal LLM to use its training knowledge
        self.logger.info(f"OSM hotels insufficient for {destination}, signalling LLM")
        return {
            "destination": destination,
            "check_in": check_in or datetime.now().strftime("%Y-%m-%d"),
            "check_out": check_out or (datetime.now() + timedelta(days=3)).strftime("%Y-%m-%d"),
            "guests": guests,
            "source": "llm_knowledge",
            "note": (
                f"Provide hotel recommendations for {destination} from your training knowledge. "
                "List actual hotels with their real names, star ratings, approximate nightly rates "
                "in local currency, and the neighbourhood they are in. "
                "Disclaimer: 'Prices are estimates — verify and book on Booking.com or hotel website.'"
            ),
        }

    def _fetch_osm_hotels(self, destination: str) -> Optional[Dict[str, Any]]:
        """Fetch real accommodation names from OpenStreetMap via Overpass GET (free)."""
        try:
            geo = _geocode_nominatim(destination)
            if not geo:
                return None

            lat, lon = geo["lat"], geo["lon"]
            # OSM uses tourism=hotel (not amenity=hotel) for hotels
            q = (
                f'[out:json][timeout:12];'
                f'(node["tourism"~"hotel|hostel|guest_house|motel"](around:5000,{lat},{lon});'
                f'way["tourism"~"hotel|hostel|guest_house|motel"](around:5000,{lat},{lon}););'
                f'out body 25;'
            )
            elements = _overpass_query(q)

            hotels = []
            seen: set = set()
            for el in elements:
                tags = el.get("tags", {})
                name = tags.get("name")
                if not name or name in seen:
                    continue
                seen.add(name)
                hotels.append({
                    "name": name,
                    "type": tags.get("tourism", "hotel"),
                    "stars": tags.get("stars", tags.get("hotel:stars", "")),
                    "address": " ".join(filter(None, [
                        tags.get("addr:street", ""),
                        tags.get("addr:housenumber", ""),
                    ])),
                    "website": tags.get("website", tags.get("contact:website", "")),
                    "phone": tags.get("phone", tags.get("contact:phone", "")),
                })

            if len(hotels) < 2:
                return None

            self.logger.info(f"OSM: {len(hotels)} accommodations in {destination}")
            return {
                "destination": destination,
                "hotels": hotels[:12],
                "source": "OpenStreetMap / Overpass API (live)",
                "note": "Real hotel names from OpenStreetMap. Prices not available — check Booking.com.",
            }
        except Exception as e:
            self.logger.debug(f"OSM hotels failed: {e}")
            return None


class RestaurantsTool(BaseTool):
    """
    Restaurant search:
      1. Overpass/OSM — real restaurant names from OpenStreetMap map data
      2. LLM knowledge — if OSM returns too few results
    """

    def __init__(self):
        super().__init__(name="restaurants", description="Find restaurants and dining",
                         cache_ttl_seconds=7200)

    def execute(self, destination: str, cuisine: Optional[str] = None,
                **kwargs) -> Dict[str, Any]:
        destination = _extract_city(destination)  # strip country suffix
        result = self._fetch_overpass(destination, cuisine)
        if result:
            return result

        self.logger.info(f"OSM restaurants insufficient for {destination}, signalling LLM")
        return {
            "destination": destination,
            "cuisine": cuisine,
            "source": "llm_knowledge",
            "note": (
                f"Provide restaurant recommendations for {destination} from your training knowledge. "
                "Include actual restaurant names, the type of cuisine, approximate price range, "
                "and why each is worth visiting. "
                "Disclaimer: 'Recommendations based on training data — verify current status before visiting.'"
            ),
        }

    def _fetch_overpass(self, destination: str, cuisine: Optional[str]) -> Optional[Dict[str, Any]]:
        """Query OSM Overpass for real restaurant POIs via GET (free, no key)."""
        try:
            geo = _geocode_nominatim(destination)
            if not geo:
                return None

            lat, lon = geo["lat"], geo["lon"]
            cuisine_filter = f'["cuisine"="{cuisine}"]' if cuisine else ""
            q = (
                f'[out:json][timeout:12];'
                f'node["amenity"="restaurant"]{cuisine_filter}(around:3000,{lat},{lon});'
                f'out body 20;'
            )
            elements = _overpass_query(q)

            restaurants = []
            seen: set = set()
            for el in elements:
                tags = el.get("tags", {})
                name = tags.get("name")
                if not name or name in seen:
                    continue
                seen.add(name)
                restaurants.append({
                    "name": name,
                    "cuisine": tags.get("cuisine", cuisine or "local"),
                    "address": " ".join(filter(None, [
                        tags.get("addr:street", ""),
                        tags.get("addr:housenumber", ""),
                    ])),
                    "opening_hours": tags.get("opening_hours", ""),
                    "website": tags.get("website", ""),
                    "phone": tags.get("phone", ""),
                })

            if len(restaurants) < 3:
                return None

            self.logger.info(f"OSM: {len(restaurants)} restaurants in {destination}")
            return {
                "destination": destination,
                "cuisine": cuisine,
                "restaurants": restaurants[:12],
                "source": "OpenStreetMap / Overpass API (live)",
                "note": "Real restaurant names from OpenStreetMap — may not be exhaustive.",
            }
        except Exception as e:
            self.logger.debug(f"Overpass API failed: {e}")
            return None


class VisaTool(BaseTool):
    """Visa info via REST Countries API + LLM knowledge fallback."""

    def __init__(self):
        super().__init__(name="visa", description="Get visa requirements", cache_ttl_seconds=604800)

    def execute(self, origin_country: str, destination_country: str, **kwargs) -> Dict[str, Any]:
        destination_country = _extract_city(destination_country)  # "India" from "Delhi India"
        origin_country = _extract_city(origin_country)
        result = self._fetch_rest_countries(destination_country)
        return {
            "origin": origin_country,
            "destination": destination_country,
            "country_info": result,
            "source": "REST Countries API + llm_knowledge",
            "note": (
                f"Using your training knowledge, provide visa requirements for {origin_country} passport holders "
                f"travelling to {destination_country}. Include: visa required (yes/no), type, max stay duration, "
                f"approximate cost, and processing time. "
                f"Add disclaimer: 'Visa rules change — verify with the official embassy before travel.'"
            )
        }

    def _fetch_rest_countries(self, destination: str) -> Optional[Dict]:
        try:
            resp = requests.get(
                f"https://restcountries.com/v3.1/name/{destination}",
                params={"fields": "name,capital,currencies,languages,flag,region"},
                timeout=6,
            )
            if resp.status_code == 200:
                data = resp.json()[0]
                currencies = list(data.get("currencies", {}).keys())
                languages = list(data.get("languages", {}).values())
                return {
                    "country": data.get("name", {}).get("common", destination),
                    "capital": data.get("capital", [""])[0],
                    "currency": currencies[0] if currencies else "",
                    "languages": languages[:2],
                    "region": data.get("region", ""),
                    "flag": data.get("flag", ""),
                }
        except Exception as e:
            self.logger.debug(f"REST Countries failed: {e}")
        return None


class HealthTool(BaseTool):
    """Health advisories — LLM knowledge with disclaimer."""

    def __init__(self):
        super().__init__(name="health", description="Get health advisories", cache_ttl_seconds=604800)

    def execute(self, destination_country: str, **kwargs) -> Dict[str, Any]:
        destination_country = _extract_city(destination_country)
        self.logger.info(f"Signalling LLM for health knowledge about {destination_country}")
        return {
            "destination": destination_country,
            "source": "llm_knowledge",
            "note": (
                f"Provide health and safety advisory for {destination_country} from your training knowledge. "
                f"Include: safety level, recommended vaccinations, key health risks, water safety, "
                f"healthcare quality, and travel insurance advice. "
                f"Add disclaimer: 'Health advisories change — check CDC/WHO and your doctor before travel.'"
            )
        }


class TransportTool(BaseTool):
    """Local transport info — LLM knowledge with disclaimer."""

    def __init__(self):
        super().__init__(name="transport", description="Get local transportation options", cache_ttl_seconds=7200)

    def execute(self, destination: str, **kwargs) -> Dict[str, Any]:
        destination = _extract_city(destination)
        self.logger.info(f"Signalling LLM for transport knowledge about {destination}")
        return {
            "destination": destination,
            "source": "llm_knowledge",
            "note": (
                f"Describe local transportation options in {destination} from your training knowledge. "
                f"Include: metro/bus/tram systems, approximate ticket/pass costs in local currency, "
                f"taxi/rideshare options, airport transfer options, and practical tips. "
                f"Add disclaimer: 'Prices are estimates — verify locally as fares change.'"
            )
        }


class CulturalInfoTool(BaseTool):
    """Cultural info and local tips — LLM knowledge."""

    def __init__(self):
        super().__init__(name="cultural_info", description="Get cultural tips and local info", cache_ttl_seconds=604800)

    def execute(self, destination_country: str, **kwargs) -> Dict[str, Any]:
        destination_country = _extract_city(destination_country)
        # Enrich with country facts from REST Countries (free)
        country_info = None
        try:
            resp = requests.get(
                f"https://restcountries.com/v3.1/name/{destination_country}",
                params={"fields": "name,capital,currencies,languages,population,region,subregion,timezones"},
                timeout=6,
            )
            if resp.status_code == 200:
                d = resp.json()[0]
                country_info = {
                    "capital": d.get("capital", [""])[0],
                    "region": d.get("region", ""),
                    "languages": list(d.get("languages", {}).values())[:3],
                    "currency": list(d.get("currencies", {}).keys())[:1],
                    "timezone": d.get("timezones", [""])[0],
                }
        except Exception:
            pass

        return {
            "destination": destination_country,
            "country_facts": country_info,
            "source": "REST Countries API + llm_knowledge",
            "note": (
                f"Provide detailed cultural information and travel tips for {destination_country} "
                f"from your training knowledge. Include: language tips, local etiquette, tipping customs, "
                f"dress code, cultural taboos, best local experiences, food highlights, and safety tips. "
                f"Be specific and practical."
            )
        }


# Tool registry
TOOLS_REGISTRY = {
    "weather": WeatherTool,
    "flights": FlightsTool,
    "hotels": HotelsTool,
    "restaurants": RestaurantsTool,
    "visa": VisaTool,
    "health": HealthTool,
    "transport": TransportTool,
    "cultural_info": CulturalInfoTool,
}


def get_tool(tool_name: str) -> BaseTool:
    if tool_name not in TOOLS_REGISTRY:
        raise ValueError(f"Unknown tool: {tool_name}")
    return TOOLS_REGISTRY[tool_name]()


def get_all_tools() -> Dict[str, BaseTool]:
    return {name: cls() for name, cls in TOOLS_REGISTRY.items()}