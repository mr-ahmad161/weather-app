"""Answers weather questions, with or without a language model.

When OPENAI_API_KEY, GOOGLE_API_KEY, GEMINI_API_KEY, or GROQ_API_KEY is set,
questions go to a LangChain agent. That agent may call search_locations and
get_weather, then write the reply. When no key is set, answer_locally reads
the city names out of the question, calls the same Open-Meteo forecast, and
writes the reply itself. Both paths return reply text, forecast cards, and
the tool steps shown in the chat.
"""

from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import AIMessage, HumanMessage

from weather import fetch_weather, search_places
from weather import WeatherServiceError

load_dotenv(Path(__file__).resolve().parent / ".env")

SYSTEM_PROMPT = """You are Sky, the weather agent inside this app.

You answer questions about current weather and the next 7 days.
You can compare more than one place.

Tools:
- search_locations: look up candidate places when a name is ambiguous.
- get_weather: live weather and a 7-day forecast for exactly one place.

Rules:
- Never invent a temperature, rain chance, wind speed, or any other weather number.
- If the number you need is not already in a tool result in this conversation, call a tool.
- For a normal question about a city, call get_weather with the place name the user used.
- When the user compares cities, call get_weather once for each city, then compare those results.
- Use search_locations only when the place name is ambiguous or get_weather could not find it.
- If a tool result says ok is false, explain that problem in plain language and ask for a clearer place. Do not guess a forecast.
- If other possible matches are listed and the user did not name a country, answer for the assumed place and mention that assumption in one short clause.
- Reply in the same language as the user. Keep Roman Urdu in Roman Urdu, Urdu script in Urdu, and English in English.
- Write a natural answer. Do not show JSON, field names, or the words "tool" and "ok".
- Temperatures in the tool results are Celsius. Mention Fahrenheit only when the user asks.
- Start with the place, the condition, and the temperature. Add feels-like, humidity, and wind when they help the answer.
- For rain questions, answer yes or no first, using precipitation and the daily rain chance.
- For comparisons, say which place is warmer, wetter, or windier, and cite the numbers you fetched.
- Offer one practical suggestion only when the data supports it, such as rain, strong wind, or high heat.
"""

# First configured key wins. LLM_MODEL=provider:model overrides the default.
PROVIDERS = (
    ("openai", ("OPENAI_API_KEY",), "gpt-4o-mini"),
    ("google_genai", ("GOOGLE_API_KEY", "GEMINI_API_KEY"), "gemini-3.7-flash"),
    ("groq", ("GROQ_API_KEY",), "openai/gpt-oss-20b"),
)


class AgentConfigError(Exception):
    """The app cannot call a model until a provider key is configured."""


def model_status() -> tuple[bool, str]:
    """Return whether a model can be called, and a short reason either way."""
    try:
        provider, model_name = resolve_llm()
    except AgentConfigError as exc:
        return False, str(exc)
    return True, f"{provider} · {model_name}"


def resolve_llm() -> tuple[str, str]:
    """Pick (provider, model) from LLM_MODEL, or from the first key that is set."""
    override = os.getenv("LLM_MODEL", "").strip()
    if override:
        if ":" not in override:
            raise AgentConfigError(
                "LLM_MODEL must look like openai:gpt-4o-mini or google_genai:gemini-3.7-flash."
            )
        provider, model_name = override.split(":", 1)
        provider = provider.strip()
        model_name = model_name.strip()
        if not provider or not model_name:
            raise AgentConfigError("LLM_MODEL is missing a provider or a model name.")
        if not _provider_ready(provider):
            raise AgentConfigError(
                f"LLM_MODEL uses {provider}, but that provider's API key is not in .env."
            )
        _alias_google_key()
        return provider, model_name

    for provider, _env_names, model_name in PROVIDERS:
        if _provider_ready(provider):
            _alias_google_key()
            return provider, model_name

    raise AgentConfigError(
        "No model API key found. Add OPENAI_API_KEY, GOOGLE_API_KEY, or GROQ_API_KEY to the .env file and restart the app."
    )


def stream_answer(history: list[dict], on_step=None) -> dict:
    """Run one agent turn. `history` includes the newest user message."""
    provider, model_name = resolve_llm()
    agent = get_agent(provider, model_name)
    cards: list[dict] = []
    steps: list[dict] = []
    pending: dict[str, dict] = {}
    final = ""

    try:
        updates = agent.stream(
            {"messages": _to_messages(history)},
            stream_mode="updates",
            config={"recursion_limit": 12},
        )
        for update in updates:
            for message in _iter_messages(update):
                final = _consume_message(message, cards, steps, pending, on_step, final)
    except Exception as exc:
        if "recursion" in str(exc).lower():
            raise AgentConfigError(
                "The agent took too many steps. Ask about one city, then compare another."
            ) from exc
        raise

    return {"reply": _clean_reply(final, cards), "cards": cards, "steps": steps}


@lru_cache(maxsize=4)
def get_agent(provider: str, model_name: str):
    model = _make_model(provider, model_name)
    return create_agent(
        model=model,
        tools=[search_locations, get_weather],
        system_prompt=SYSTEM_PROMPT,
        name="Sky",
    )


def public_error(exc: Exception) -> str:
    """Turn a model failure into a short message, with any API key removed."""
    text = " ".join(str(exc).split())
    for env_name in (
        "OPENAI_API_KEY",
        "GOOGLE_API_KEY",
        "GEMINI_API_KEY",
        "GROQ_API_KEY",
        "ANTHROPIC_API_KEY",
    ):
        secret = os.getenv(env_name)
        if secret:
            text = text.replace(secret, "••••")
    lowered = text.lower()
    if "api key" in lowered or "authentication" in lowered or "401" in lowered:
        return "The model rejected the API key. Check the key in .env, save the file, and restart the app."
    if "model" in lowered and ("not found" in lowered or "does not exist" in lowered or "404" in lowered):
        return (
            "That model name was rejected. Set LLM_MODEL in .env to a model your key can use, "
            "for example openai:gpt-4o-mini or google_genai:gemini-3.7-flash, then restart."
        )
    if not text:
        return "The agent could not finish that request. Try again in a moment."
    return text[:400]


@tool
def search_locations(query: str) -> str:
    """Find places that match a name when it could refer to more than one city.

    Args:
        query: Place name to search, such as "Springfield" or "Lahore".
    """
    try:
        matches = search_places(query)
    except WeatherServiceError as exc:
        return json.dumps({"ok": False, "error": str(exc)})
    except Exception as exc:
        return json.dumps({"ok": False, "error": f"Location search failed: {exc}"})

    if not matches:
        cleaned = " ".join(str(query).split())
        return json.dumps({"ok": False, "error": f"No places found for '{cleaned}'."})

    brief = [
        {
            "label": place["label"],
            "latitude": round(place["latitude"], 2),
            "longitude": round(place["longitude"], 2),
            "population": place.get("population"),
        }
        for place in matches
    ]
    return json.dumps({"ok": True, "matches": brief})


@tool
def get_weather(place: str) -> str:
    """Get the current weather and 7-day forecast for one place. Call once per city.

    Args:
        place: City or area, with a country when you know it. Example: "Lahore, Pakistan".
    """
    try:
        payload = fetch_weather(place)
    except Exception as exc:
        return json.dumps({"ok": False, "error": f"Weather lookup failed: {exc}"})

    if not payload.get("ok"):
        return json.dumps({"ok": False, "error": payload.get("error") or "Weather lookup failed."})

    card = payload["card"]
    return json.dumps(
        {
            "ok": True,
            "assumed_place": card["place"]["label"],
            "other_matches": payload.get("alternatives") or [],
            "report": payload["report"],
            "card": card,
        },
        ensure_ascii=False,
    )


def _provider_ready(provider: str) -> bool:
    for name, env_names, _model in PROVIDERS:
        if name == provider:
            return any(os.getenv(env_name) for env_name in env_names)
    return False


def _alias_google_key() -> None:
    if not os.getenv("GOOGLE_API_KEY") and os.getenv("GEMINI_API_KEY"):
        os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]


def _make_model(provider: str, model_name: str):
    model_id = f"{provider}:{model_name}"
    try:
        return init_chat_model(model_id, temperature=0)
    except TypeError:
        return init_chat_model(model_id)


def _to_messages(history: list[dict]) -> list:
    converted = []
    for item in history[-10:]:
        content = str(item.get("content") or "").strip()
        if not content:
            continue
        if item.get("role") == "assistant":
            converted.append(AIMessage(content=content))
        else:
            converted.append(HumanMessage(content=content))
    while converted and isinstance(converted[0], AIMessage):
        converted.pop(0)
    if not converted:
        raise AgentConfigError("Type a weather question to start.")
    return converted


def _iter_messages(update):
    if not isinstance(update, dict):
        return
    for payload in update.values():
        messages = None
        if isinstance(payload, dict):
            messages = payload.get("messages")
        elif hasattr(payload, "messages"):
            messages = payload.messages
        if messages is None:
            continue
        if not isinstance(messages, list):
            messages = [messages]
        yield from messages


def _consume_message(message, cards, steps, pending, on_step, final: str) -> str:
    """Record a tool call, attach its result to a card, or keep the latest reply."""
    tool_calls = getattr(message, "tool_calls", None) or []
    if tool_calls:
        for call in tool_calls:
            step = {
                "id": call.get("id"),
                "tool": call.get("name") or "tool",
                "detail": _call_detail(call),
                "ok": None,
            }
            steps.append(step)
            if step["id"]:
                pending[step["id"]] = step
            if on_step:
                on_step(step)
        return final

    tool_call_id = getattr(message, "tool_call_id", None)
    if tool_call_id:
        payload = _parse_payload(getattr(message, "content", ""))
        step = pending.get(tool_call_id)
        if payload and payload.get("ok") and payload.get("card"):
            cards.append(payload["card"])
            if step is not None:
                step["ok"] = True
                if payload.get("assumed_place"):
                    step["detail"] = payload["assumed_place"]
        elif step is not None:
            step["ok"] = False
            if payload and payload.get("error"):
                step["detail"] = payload["error"]
        return final

    text = _content_text(getattr(message, "content", "")).strip()
    return text or final


def _parse_payload(content) -> dict | None:
    text = _content_text(content).strip()
    if not text:
        return None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _content_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(str(block.get("text") or block.get("content") or ""))
        return "\n".join(part for part in parts if part)
    return str(content or "")


def _call_detail(call: dict) -> str:
    args = call.get("args") or {}
    for key in ("place", "query", "city", "location"):
        if args.get(key):
            return str(args[key])
    return call.get("name") or "tool"


def _clean_reply(text: str, cards: list[dict]) -> str:
    stripped = text.strip()
    if not stripped or stripped.startswith("{") or stripped.startswith("```"):
        return _fallback_reply(cards)
    return stripped


def answer_question(history: list[dict], on_step=None) -> dict:
    """Answer with LangChain when a free or paid key exists, otherwise with the local agent.

    The local agent uses the same Open-Meteo tools and does not need an API key.
    """
    ready, _detail = model_status()
    if ready:
        return stream_answer(history, on_step=on_step)
    question = ""
    for item in reversed(history):
        if item.get("role") == "user" and item.get("content"):
            question = str(item["content"])
            break
    return answer_locally(question, on_step=on_step)


def answer_locally(question: str, on_step=None) -> dict:
    """Answer from Open-Meteo without calling a language model."""
    places = places_in_question(question)
    if not places:
        return {
            "reply": _no_place_reply(question),
            "cards": [],
            "steps": [],
        }

    cards = []
    steps = []
    errors = []
    for place in places[:3]:
        step = {"tool": "get_weather", "detail": place, "ok": None}
        steps.append(step)
        if on_step:
            on_step(step)
        result = fetch_weather(place)
        if result.get("ok"):
            card = result["card"]
            cards.append(card)
            step["ok"] = True
            step["detail"] = card["place"]["label"]
        else:
            step["ok"] = False
            step["detail"] = result.get("error") or place
            errors.append(result.get("error") or f"No forecast for {place}.")

    if not cards:
        return {"reply": errors[0], "cards": [], "steps": steps}
    return {
        "reply": _local_reply(question, cards, errors),
        "cards": cards,
        "steps": steps,
    }


def places_in_question(question: str) -> list[str]:
    """Extract up to two place names from an English or Roman Urdu question.

    Comparison questions are read as "compare ... in A and B". Otherwise a
    known city name is preferred, then phrases such as "in London" or
    "Lahore mein".
    """
    text = " ".join(str(question).replace("?", " ").split())
    lowered = text.lower()
    compare = re.search(r"compare (?:the )?weather in (.+?) and (.+)$", text, flags=re.IGNORECASE)
    if compare:
        return [_clean_place(compare.group(1)), _clean_place(compare.group(2))]

    known = _known_places(lowered)
    if known:
        return known

    for pattern in (
        r"(?:in|at)\s+(.+?)(?:\s+this\b|\s+today\b|\s+tomorrow\b|\s+tonight\b|$)",
        r"^(.+?)\s+mein\b",
        r"^(.+?)\s+ka mausam\b",
    ):
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return [_clean_place(match.group(1))]
    return []


_KNOWN_PLACES = (
    "lahore",
    "karachi",
    "islamabad",
    "rawalpindi",
    "murree",
    "peshawar",
    "quetta",
    "multan",
    "faisalabad",
    "hyderabad",
    "london",
    "dubai",
    "paris",
    "new york",
    "tokyo",
    "delhi",
    "mumbai",
    "istanbul",
)


def _known_places(lowered: str) -> list[str]:
    """Find known city names, longest first so "new york" is not read as nothing."""
    hits: list[tuple[int, str]] = []
    occupied: list[tuple[int, int]] = []
    for name in sorted(_KNOWN_PLACES, key=len, reverse=True):
        for match in re.finditer(rf"\b{re.escape(name)}\b", lowered):
            span = match.span()
            if any(not (span[1] <= start or span[0] >= end) for start, end in occupied):
                continue
            occupied.append(span)
            label = "New York" if name == "new york" else name.title()
            hits.append((span[0], label))
    hits.sort()
    return [label for _start, label in hits]


def _clean_place(value: str) -> str:
    return " ".join(value.replace(",", " ").split()).strip(" .")


def _local_reply(question: str, cards: list[dict], errors: list[str]) -> str:
    """Choose a current, rain, or comparison sentence from the fetched cards."""
    roman = _uses_roman_urdu(question)
    if len(cards) > 1 or "compare" in question.lower():
        reply = _compare_reply(cards, roman)
    elif _asks_about_rain(question):
        reply = _rain_reply(cards[0], question, roman)
    else:
        reply = _current_reply(cards[0], roman)
    if errors:
        reply = f"{reply} {' '.join(errors)}"
    return reply


def _uses_roman_urdu(question: str) -> bool:
    """True for Urdu script or common Roman Urdu weather words."""
    if re.search(r"[\u0600-\u06FF]", question):
        return True
    return bool(re.search(r"\b(mein|mausam|kaisa|kaisi|barish|hoga|hogi|kya)\b", question.lower()))


def _asks_about_rain(question: str) -> bool:
    return bool(re.search(r"\b(rain|raining|barish|wet)\b", question.lower()))


def _no_place_reply(question: str) -> str:
    if _uses_roman_urdu(question):
        return "Sheher ka naam likho, jaise Lahore ya Karachi."
    return "Tell me a city, for example Lahore or Karachi."


def _current_reply(card: dict, roman: bool) -> str:
    current = card["current"]
    place = card["place"]["label"]
    if roman:
        return (
            f"{place} mein abhi {current['temperature']}°C hai aur mausam {current['condition'].lower()} hai. "
            f"Mehsoos {current['apparent_temperature']}°C ho raha hai. "
            f"Humidity {current['humidity']}% hai aur hawa {current['wind_speed']} km/h hai."
        )
    return (
        f"{place} is {current['temperature']}°C and {current['condition'].lower()}, "
        f"feeling like {current['apparent_temperature']}°C. "
        f"Humidity is {current['humidity']}% and the wind is {_wind_phrase(current)}."
    )


def _rain_reply(card: dict, question: str, roman: bool) -> str:
    current = card["current"]
    place = card["place"]["name"]
    wettest = max(card["daily"], key=lambda day: day.get("precipitation_probability") or 0)
    chance = wettest.get("precipitation_probability") or 0
    raining = _is_raining(current)
    weekly = bool(re.search(r"\b(week|hafte|hafta)\b", question.lower()))
    if roman and weekly:
        if chance >= 30:
            return (
                f"{place} mein agle 7 din mein {wettest['weekday']} ko barish ka chance {chance}% hai. "
                f"Abhi {current['temperature']}°C hai aur mausam {current['condition'].lower()} hai."
            )
        return (
            f"{place} mein agle 7 din zyada tar khushk lag rahe hain. "
            f"Sab se zyada chance {chance}% hai, {wettest['weekday']} ko. Abhi {current['temperature']}°C hai."
        )
    if roman:
        if raining:
            return f"Haan, {place} mein abhi barish ho rahi hai. Temperature {current['temperature']}°C hai."
        return (
            f"Nahi, {place} mein abhi barish nahi ho rahi. "
            f"Aaj barish ka chance {card['daily'][0]['precipitation_probability']}% hai "
            f"aur temperature {current['temperature']}°C hai."
        )
    if weekly:
        if chance >= 30:
            return (
                f"Rain is possible in {place} this week, mainly on {wettest['weekday']} "
                f"when the chance is {chance}%. It is {current['temperature']}°C and "
                f"{current['condition'].lower()} right now."
            )
        return (
            f"The next 7 days in {place} look mostly dry. The highest rain chance is {chance}% "
            f"on {wettest['weekday']}. Right now it is {current['temperature']}°C and "
            f"{current['condition'].lower()}."
        )
    if raining:
        return (
            f"Yes. It is raining in {place} right now. "
            f"The temperature is {current['temperature']}°C and conditions are {current['condition'].lower()}."
        )
    today = card["daily"][0]
    return (
        f"No. It is not raining in {place} right now. "
        f"Today's rain chance is {today['precipitation_probability']}%, "
        f"and it is {current['temperature']}°C and {current['condition'].lower()}."
    )


def _compare_reply(cards: list[dict], roman: bool) -> str:
    warmer = max(cards, key=lambda card: card["current"]["temperature"])
    cooler = min(cards, key=lambda card: card["current"]["temperature"])
    parts = []
    for card in cards:
        current = card["current"]
        if roman:
            parts.append(
                f"{card['place']['name']} mein {current['temperature']}°C hai aur mausam {current['condition'].lower()} hai"
            )
        else:
            parts.append(
                f"{card['place']['label']} is {current['temperature']}°C and {current['condition'].lower()}"
            )
    if roman:
        gap = warmer["current"]["temperature"] - cooler["current"]["temperature"]
        return (
            f"{'. '.join(parts)}. "
            f"{warmer['place']['name']} {gap}°C zyada garm hai {cooler['place']['name']} se."
        )
    gap = warmer["current"]["temperature"] - cooler["current"]["temperature"]
    if gap == 0:
        ending = "Both places are the same temperature."
    else:
        ending = f"{warmer['place']['name']} is warmer by {gap}°C."
    return f"{'. '.join(parts)}. {ending}"


def _is_raining(current: dict) -> bool:
    condition = str(current.get("condition") or "").lower()
    precipitation = current.get("precipitation") or 0
    return precipitation > 0 or any(word in condition for word in ("rain", "drizzle", "thunder", "shower"))


def _wind_phrase(current: dict) -> str:
    speed = current.get("wind_speed")
    direction = current.get("wind_direction")
    if direction:
        return f"{speed} km/h from {direction}"
    return f"{speed} km/h"


def _fallback_reply(cards: list[dict]) -> str:
    """Used when the model returns empty text or raw JSON instead of a sentence."""
    if not cards:
        return "I couldn't find weather details for that. Try a city and country, such as Lahore, Pakistan."
    sentences = []
    for card in cards:
        current = card["current"]
        sentences.append(
            f"{card['place']['label']} is {current['temperature']}°C and {str(current['condition']).lower()}, "
            f"feeling like {current['apparent_temperature']}°C."
        )
    return " ".join(sentences)
