"""Live weather lookups through Open-Meteo.

Open-Meteo is the weather source for this app: current conditions, a short
hourly outlook, and a 7-day forecast, with no API key for non-commercial use.
https://open-meteo.com/en/docs
"""

from __future__ import annotations

import time
from datetime import datetime

import httpx

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

_CLIENT = httpx.Client(
    timeout=httpx.Timeout(20.0, connect=8.0),
    headers={"User-Agent": "sky-agent/1.0 (educational langchain weather app)"},
    follow_redirects=True,
)

_CONDITIONS = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Rime fog",
    51: "Light drizzle",
    53: "Drizzle",
    55: "Heavy drizzle",
    56: "Light freezing drizzle",
    57: "Freezing drizzle",
    61: "Light rain",
    63: "Rain",
    65: "Heavy rain",
    66: "Light freezing rain",
    67: "Freezing rain",
    71: "Light snow",
    73: "Snow",
    75: "Heavy snow",
    77: "Snow grains",
    80: "Light rain showers",
    81: "Rain showers",
    82: "Heavy rain showers",
    85: "Snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with hail",
    99: "Thunderstorm with hail",
}


class WeatherServiceError(Exception):
    """The location search or forecast request could not be completed."""


def condition_for(code: int | None) -> str:
    """Map a WMO weather code to a short English condition."""
    if code is None:
        return "Unknown conditions"
    return _CONDITIONS.get(int(code), "Unknown conditions")


def emoji_for(code: int | None, is_day: bool = True) -> str:
    """Pick an icon for a WMO code, using a moon for clear night hours."""
    if code is None:
        return "🌡️"
    code = int(code)
    if code == 0:
        return "☀️" if is_day else "🌙"
    if code in (1, 2):
        return "⛅" if is_day else "☁️"
    if code == 3:
        return "☁️"
    if code in (45, 48):
        return "🌫️"
    if 51 <= code <= 67 or 80 <= code <= 82:
        return "🌧️"
    if 71 <= code <= 77 or 85 <= code <= 86:
        return "❄️"
    if 95 <= code <= 99:
        return "⛈️"
    return "🌡️"


def theme_for(code: int | None, is_day: bool = True) -> str:
    """Return the card theme name used by the stylesheet gradients."""
    if code is None:
        return "cloud"
    code = int(code)
    if 95 <= code <= 99:
        return "storm"
    if 71 <= code <= 77 or 85 <= code <= 86:
        return "snow"
    if 51 <= code <= 67 or 80 <= code <= 82:
        return "rain"
    if code in (45, 48):
        return "fog"
    if code == 0:
        return "clear-day" if is_day else "clear-night"
    if code in (1, 2):
        return "partly-day" if is_day else "partly-night"
    return "cloud"


def compass(degrees: float | None) -> str | None:
    """Convert a wind direction in degrees to an 8-point compass label."""
    if degrees is None:
        return None
    names = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    return names[round(float(degrees) / 45) % 8]


def search_places(query: str, count: int = 5) -> list[dict]:
    """Return matching places for a name. Raises WeatherServiceError on failure."""
    name = " ".join(str(query).split())
    if not name:
        raise WeatherServiceError("Type a city or place name first.")
    if len(name) > 80:
        raise WeatherServiceError("That place name is too long.")

    try:
        payload = _read_json(
            GEOCODE_URL,
            {"name": name, "count": count, "language": "en", "format": "json"},
        )
    except httpx.HTTPError as exc:
        raise WeatherServiceError(
            "The location search is unavailable right now. Try again in a moment."
        ) from exc
    except ValueError as exc:
        raise WeatherServiceError(
            "The location search returned a response that could not be read."
        ) from exc

    places = []
    for item in payload.get("results") or []:
        if "latitude" not in item or "longitude" not in item:
            continue
        places.append(_normalize_place(item))
    return places


def fetch_weather(place: str) -> dict:
    """Look up one place and return a result the agent and the UI can both use.

    Success: {"ok": True, "report": str, "card": dict, "alternatives": list}
    Failure: {"ok": False, "error": str}
    """
    try:
        matches = search_places(place)
    except WeatherServiceError as exc:
        return {"ok": False, "error": str(exc)}

    if not matches:
        cleaned = " ".join(str(place).split())
        return {
            "ok": False,
            "error": f"I couldn't find a place named '{cleaned}'. Try adding a country, such as 'Lahore, Pakistan'.",
        }

    chosen = matches[0]
    try:
        card = _forecast_card(chosen)
    except WeatherServiceError as exc:
        return {"ok": False, "error": str(exc)}

    alternatives = [item["label"] for item in matches[1:4]]
    card["alternatives"] = alternatives
    return {
        "ok": True,
        "report": build_report(card),
        "card": card,
        "alternatives": alternatives,
    }


def build_report(card: dict) -> str:
    """Plain-text forecast the model reads before it writes an answer."""
    current = card["current"]
    place = card["place"]
    wind = f"{current['wind_speed']} km/h"
    if current.get("wind_direction"):
        wind += f" from {current['wind_direction']}"
    if current.get("wind_gusts") is not None:
        wind += f", gusts {current['wind_gusts']} km/h"

    lines = [
        f"Place: {place['label']}",
        f"Coordinates: {place['latitude']}, {place['longitude']}",
        f"Timezone: {place['timezone']}",
        f"Observed local time: {current['time']}",
        "Units: °C, km/h, millimetres.",
        "",
        "Current:",
        f"- Condition: {current['condition']}",
        f"- Temperature: {current['temperature']}°C",
        f"- Feels like: {current['apparent_temperature']}°C",
        f"- Humidity: {current['humidity']}%",
        f"- Wind: {wind}",
        f"- Precipitation: {current['precipitation']} mm",
        f"- Cloud cover: {current['cloud_cover']}%",
        f"- Pressure: {current['pressure']} hPa",
        f"- Visibility: {current['visibility_km']} km",
        f"- Daylight: {'yes' if current['is_day'] else 'no'}",
    ]

    today = card["daily"][0] if card["daily"] else None
    if today:
        lines.extend(
            [
                f"- Today's rain chance: {today['precipitation_probability']}%",
                f"- Today's rain total: {today['precipitation_sum']} mm",
                f"- UV index: {today['uv_index']}",
                f"- Sunrise: {today['sunrise']}",
                f"- Sunset: {today['sunset']}",
            ]
        )

    lines.extend(["", "Next hours:"])
    for hour in card["hourly"]:
        lines.append(
            f"- {hour['label']}: {hour['temperature']}°C, {hour['condition']}, "
            f"rain chance {hour['precipitation_probability']}%"
        )

    lines.extend(["", "7-day forecast:"])
    for day in card["daily"]:
        lines.append(
            f"- {day['date']} ({day['weekday']}): {day['temp_min']} to {day['temp_max']}°C, "
            f"{day['condition']}, rain chance {day['precipitation_probability']}%, "
            f"rain {day['precipitation_sum']} mm, wind up to {day['wind_speed_max']} km/h"
        )

    if card.get("alternatives"):
        lines.extend(["", "Other possible matches: " + "; ".join(card["alternatives"])])
    else:
        lines.extend(["", "Other possible matches: none"])
    return "\n".join(lines)


def _read_json(url: str, params: dict) -> dict:
    """GET JSON, and try once more if the service blips."""
    last_error: Exception | None = None
    for attempt in range(2):
        try:
            response = _CLIENT.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPError, ValueError) as exc:
            last_error = exc
            if attempt == 0:
                time.sleep(0.5)
    assert last_error is not None
    raise last_error


def _normalize_place(item: dict) -> dict:
    name = item.get("name") or "Unknown place"
    admin1 = (item.get("admin1") or "").strip()
    country = (item.get("country") or "").strip()
    parts = [name]
    if admin1 and admin1 != name:
        parts.append(admin1)
    if country:
        parts.append(country)
    return {
        "name": name,
        "admin1": admin1,
        "country": country,
        "country_code": item.get("country_code") or "",
        "label": ", ".join(parts),
        "latitude": float(item["latitude"]),
        "longitude": float(item["longitude"]),
        "timezone": item.get("timezone") or "UTC",
        "population": item.get("population"),
    }


def _forecast_card(place: dict) -> dict:
    """Request the Open-Meteo forecast and shape it into one UI card."""
    params = {
        "latitude": place["latitude"],
        "longitude": place["longitude"],
        "timezone": "auto",
        "forecast_days": 7,
        "wind_speed_unit": "kmh",
        "precipitation_unit": "mm",
        "current": ",".join(
            [
                "temperature_2m",
                "relative_humidity_2m",
                "apparent_temperature",
                "precipitation",
                "weather_code",
                "wind_speed_10m",
                "wind_direction_10m",
                "wind_gusts_10m",
                "cloud_cover",
                "pressure_msl",
                "is_day",
                "visibility",
            ]
        ),
        "hourly": ",".join(
            [
                "temperature_2m",
                "precipitation_probability",
                "weather_code",
                "wind_speed_10m",
            ]
        ),
        "daily": ",".join(
            [
                "weather_code",
                "temperature_2m_max",
                "temperature_2m_min",
                "precipitation_sum",
                "precipitation_probability_max",
                "sunrise",
                "sunset",
                "uv_index_max",
                "wind_speed_10m_max",
            ]
        ),
    }
    try:
        payload = _read_json(FORECAST_URL, params)
    except httpx.HTTPError as exc:
        raise WeatherServiceError(
            "The forecast service is unavailable right now. Try again in a moment."
        ) from exc
    except ValueError as exc:
        raise WeatherServiceError(
            "The forecast service returned a response that could not be read."
        ) from exc

    current = payload.get("current") or {}
    if "temperature_2m" not in current:
        raise WeatherServiceError("The forecast did not include current weather.")

    code = _as_int(current.get("weather_code"))
    is_day = bool(current.get("is_day", 1))
    daily = _daily(payload.get("daily") or {})
    hourly = _hourly(payload.get("hourly") or {}, current.get("time") or "", daily, is_day)
    visibility = current.get("visibility")

    return {
        "place": {
            "name": place["name"],
            "admin1": place["admin1"],
            "country": place["country"],
            "label": place["label"],
            "latitude": round(place["latitude"], 2),
            "longitude": round(place["longitude"], 2),
            "timezone": (payload.get("timezone") or place["timezone"]),
        },
        "theme": theme_for(code, is_day),
        "current": {
            "time": current.get("time") or "",
            "temperature": _num(current.get("temperature_2m")),
            "apparent_temperature": _num(current.get("apparent_temperature")),
            "humidity": _num(current.get("relative_humidity_2m")),
            "precipitation": _num(current.get("precipitation"), 1),
            "weather_code": code,
            "condition": condition_for(code),
            "emoji": emoji_for(code, is_day),
            "wind_speed": _num(current.get("wind_speed_10m")),
            "wind_direction": compass(current.get("wind_direction_10m")),
            "wind_gusts": _num(current.get("wind_gusts_10m")),
            "cloud_cover": _num(current.get("cloud_cover")),
            "pressure": _num(current.get("pressure_msl")),
            "is_day": is_day,
            "visibility_km": None if visibility is None else _num(float(visibility) / 1000, 1),
        },
        "hourly": hourly,
        "daily": daily,
        "alternatives": [],
    }


def hour_is_day(stamp: str, daily: list[dict], fallback: bool = True) -> bool:
    """True when the hour falls between that day's sunrise and sunset."""
    text = str(stamp)
    if len(text) < 16:
        return fallback
    date, clock = text[:10], text[11:16]
    for day in daily:
        if day.get("date") == date and day.get("sunrise") and day.get("sunset"):
            return str(day["sunrise"]) <= clock < str(day["sunset"])
    return fallback


def _hourly(block: dict, current_time: str, daily: list[dict], fallback_is_day: bool) -> list[dict]:
    """Keep the next eight hours, starting from the current local hour."""
    times = block.get("time") or []
    temperatures = block.get("temperature_2m") or []
    probabilities = block.get("precipitation_probability") or []
    codes = block.get("weather_code") or []
    if not times:
        return []

    hour_key = str(current_time)[:13]
    start = 0
    for index, value in enumerate(times):
        if str(value)[:13] >= hour_key:
            start = index
            break

    hours = []
    for index in range(start, min(start + 8, len(times))):
        code = _as_int(_at(codes, index))
        stamp = times[index]
        hours.append(
            {
                "time": stamp,
                "label": _hour_label(stamp),
                "temperature": _num(_at(temperatures, index)),
                "precipitation_probability": _num(_at(probabilities, index)),
                "weather_code": code,
                "condition": condition_for(code),
                "emoji": emoji_for(code, hour_is_day(stamp, daily, fallback_is_day)),
            }
        )
    return hours


def _daily(block: dict) -> list[dict]:
    """Shape the next seven daily rows, including sunrise and sunset clocks."""
    dates = block.get("time") or []
    days = []
    for index, date in enumerate(dates[:7]):
        code = _as_int(_at(block.get("weather_code") or [], index))
        days.append(
            {
                "date": date,
                "weekday": _weekday(date),
                "weather_code": code,
                "condition": condition_for(code),
                "emoji": emoji_for(code, True),
                "temp_max": _num(_at(block.get("temperature_2m_max") or [], index)),
                "temp_min": _num(_at(block.get("temperature_2m_min") or [], index)),
                "precipitation_sum": _num(_at(block.get("precipitation_sum") or [], index), 1),
                "precipitation_probability": _num(
                    _at(block.get("precipitation_probability_max") or [], index)
                ),
                "sunrise": _clock(_at(block.get("sunrise") or [], index)),
                "sunset": _clock(_at(block.get("sunset") or [], index)),
                "uv_index": _num(_at(block.get("uv_index_max") or [], index), 1),
                "wind_speed_max": _num(_at(block.get("wind_speed_10m_max") or [], index)),
            }
        )
    return days


def _at(values: list, index: int):
    if index < len(values):
        return values[index]
    return None


def _as_int(value) -> int | None:
    if value is None:
        return None
    return int(value)


def _num(value, digits: int = 0):
    if value is None:
        return None
    number = round(float(value), digits)
    if digits == 0:
        return int(number)
    return number


def _hour_label(iso_time: str) -> str:
    try:
        parsed = datetime.strptime(str(iso_time)[:16], "%Y-%m-%dT%H:%M")
    except ValueError:
        return str(iso_time)
    return parsed.strftime("%I %p").lstrip("0")


def _weekday(iso_date: str) -> str:
    try:
        parsed = datetime.strptime(str(iso_date)[:10], "%Y-%m-%d")
    except ValueError:
        return str(iso_date)
    return parsed.strftime("%a")


def _clock(value) -> str | None:
    if not value:
        return None
    text = str(value)
    if "T" in text:
        return text.split("T", 1)[1][:5]
    return text[:5]
