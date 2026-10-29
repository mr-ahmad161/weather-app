# Sky

Sky is a weather agent. You search a city or ask a question in plain language, and the app answers from a live forecast. It shows the current conditions, the next few hours, and the next seven days. It can compare two cities and reply in English or Roman Urdu.

Weather data comes from [Open-Meteo](https://open-meteo.com/). No weather API key is required. Chat works without a paid model key as well. If you later add an OpenAI, Google, or Groq key, the same questions are answered by a LangChain agent instead.

## What you can do

- Search any city, or pick Lahore, Karachi, Islamabad, Murree, London, or Dubai.
- Read current temperature, feels-like, humidity, wind, rain chance, UV, hourly outlook, and a 7-day forecast.
- Ask questions such as whether it is raining, what the week looks like, or which of two cities is warmer.
- Ask in Roman Urdu, for example `Lahore mein mausam kaisa hai?`

## How it works

1. `app.py` is the Streamlit page. It loads the selected city's card and sends chat questions to the agent.
2. `agent.py` decides how to answer.
   - With a model key, a LangChain agent calls `search_locations` and `get_weather`, then writes the reply.
   - Without a key, the app reads the city names from the question, calls the same forecast, and writes the reply itself.
3. `weather.py` geocodes the place and requests the Open-Meteo forecast. Temperatures are Celsius, wind is km/h, and rain is millimetres. A failed request is retried once.
4. `present.py` turns that forecast into the cards on the page. The card colour follows the condition, such as clear, rain, snow, or storm.

The agent is instructed not to invent weather numbers. If a place cannot be found, the app says so instead of guessing.

## Requirements

- Python 3.12 or newer
- [uv](https://docs.astral.sh/uv/)

## Setup

From this folder:

```powershell
uv sync
Copy-Item .env.example .env
```

You can leave `.env` empty. The app runs with Open-Meteo only.

A model key is optional. Put one key in `.env` if you want LangChain to write the answers:

```
OPENAI_API_KEY=your_key_here
```

Google and Groq work the same way with `GOOGLE_API_KEY` or `GROQ_API_KEY`. To choose the model, set `LLM_MODEL` as `provider:model`, for example:

```
LLM_MODEL=openai:gpt-4o-mini
```

Restart the app after changing `.env`. Do not commit `.env`.

## Run

```powershell
uv run streamlit run app.py
```

Open the local URL Streamlit prints, usually `http://localhost:8501`.

## Example questions

- What is the weather in Lahore, Pakistan?
- Tell me the current temperature in London.
- Is it raining in Karachi?
- Compare the weather in Lahore and Islamabad.
- Will it rain in Murree this week?
- Lahore mein mausam kaisa hai?

## Project layout

| Path | Purpose |
| --- | --- |
| `app.py` | Streamlit page: city search, forecast card, and chat |
| `agent.py` | LangChain tools, optional model, and the built-in answers |
| `weather.py` | Open-Meteo geocoding and forecast |
| `present.py` | Header, tool steps, and forecast card HTML |
| `assets/styles.css` | Page layout and card styles |
| `tests/test_weather.py` | Offline tests for city parsing and weather labels |
| `requirements.txt` | Pinned dependencies for pip |
| `.env.example` | Optional model settings, with no secrets |

## Tests

```powershell
uv run python -m unittest discover -s tests -v
```

The tests do not call the weather service. They check that sample questions resolve to the right cities, and that condition labels, compass directions, and blank input behave as expected.

## Install with pip

uv is the intended setup. If you need pip instead:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```
