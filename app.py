"""Streamlit interface for the Sky weather agent.

The page does two things. The top half shows a live forecast card for the
city the user searches or selects. The bottom half is a chat: suggested
questions and the text box both go through the same agent. Forecasts come
from Open-Meteo and do not need an API key. A model key is optional; without
one the agent still answers from the same weather data.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from agent import AgentConfigError, answer_question, public_error
from present import page_header, render_cards, render_steps
from weather import fetch_weather

load_dotenv(Path(__file__).resolve().parent / ".env")

# Quick picks shown as pills, and the questions offered before the first message.
CITIES = ["Lahore", "Karachi", "Islamabad", "Murree", "London", "Dubai"]
PROMPTS = [
    "What is the weather in Lahore, Pakistan?",
    "Is it raining in Karachi?",
    "Compare the weather in Lahore and Islamabad.",
    "Will it rain in Murree this week?",
    "Tell me the current temperature in London.",
    "Lahore mein mausam kaisa hai?",
]

st.set_page_config(page_title="Sky", page_icon="⛅", layout="centered")
st.html((Path(__file__).resolve().parent / "assets" / "styles.css").read_text(encoding="utf-8"))

# Session keys survive Streamlit reruns. "queued" is popped immediately so a
# suggested question runs once, on the next pass, instead of on every rerun.
if "messages" not in st.session_state:
    st.session_state.messages = []
if "place" not in st.session_state:
    st.session_state.place = "Lahore"

pending = st.session_state.pop("queued", None)


def draw_message(message: dict) -> None:
    """Render one stored chat turn, including any forecast cards on replies."""
    avatar = "⛅" if message["role"] == "assistant" else "🙂"
    with st.chat_message(message["role"], avatar=avatar):
        if message["role"] == "assistant":
            draw_assistant(message)
        else:
            st.markdown(message["content"])


def draw_assistant(message: dict) -> None:
    """Show tool steps, forecast cards, then the written answer or an error."""
    if message.get("steps"):
        st.html(render_steps(message["steps"]))
    if message.get("cards"):
        st.html(render_cards(message["cards"]))
    if message.get("error"):
        st.error(message["content"])
    else:
        st.markdown(message["content"])


def remember(role: str, content: str, **extra) -> dict:
    """Append a chat turn to session state and return it for immediate display."""
    message = {"role": role, "content": content, **extra}
    st.session_state.messages.append(message)
    return message


def run_question(text: str) -> None:
    """Send one question to the agent and store both the question and the reply.

    Blank input is ignored. Long input is cut at 500 characters so a pasted
    page cannot be forwarded as a prompt. Failed turns are stored with
    error=True and are left out of later model history.
    """
    cleaned = " ".join(text.split())
    if not cleaned:
        return
    cleaned = cleaned[:500]
    draw_message(remember("user", cleaned))
    history = [
        {"role": item["role"], "content": item["content"]}
        for item in st.session_state.messages
        if item.get("content") and not item.get("error")
    ]

    with st.chat_message("assistant", avatar="⛅"):
        try:
            with st.status("Checking the forecast...", expanded=True) as status:

                def on_step(step: dict) -> None:
                    status.write(f"{step['tool']} · {step['detail']}")

                result = answer_question(history, on_step=on_step)
                status.update(label="Answer ready", state="complete", expanded=False)
        except AgentConfigError as exc:
            reply = str(exc)
            st.error(reply)
            remember("assistant", reply, cards=[], steps=[], error=True)
            return
        except Exception as exc:
            reply = public_error(exc)
            st.error(reply)
            remember("assistant", reply, cards=[], steps=[], error=True)
            return

        answer = {
            "role": "assistant",
            "content": result["reply"],
            "cards": result["cards"],
            "steps": result["steps"],
        }
        draw_assistant(answer)
        st.session_state.messages.append(answer)


@st.cache_data(ttl=600, show_spinner=False)
def load_place(name: str) -> dict:
    """Fetch one city forecast and reuse it for ten minutes."""
    return fetch_weather(name)


st.html(page_header())

if "city_pill" not in st.session_state:
    st.session_state.city_pill = "Lahore"

with st.form("lookup", border=False):
    field, action = st.columns([5, 1], vertical_alignment="bottom")
    city = field.text_input(
        "City",
        placeholder="Search a city",
        label_visibility="collapsed",
    )
    submitted = action.form_submit_button("Search", use_container_width=True)

# A search sets the pill before the widget is created. Streamlit rejects a
# change to a widget key after that widget already exists in the same run.
if submitted and city.strip():
    place = " ".join(city.split())
    st.session_state.place = place
    st.session_state.city_pill = next(
        (name for name in CITIES if name.lower() == place.lower()),
        None,
    )

choice = st.pills(
    "Cities",
    CITIES,
    key="city_pill",
    label_visibility="collapsed",
    width="stretch",
    wrap=True,
)

# Ignore the pill on the same run as a search, or the pill would overwrite it.
if not submitted and choice and choice != st.session_state.get("shown_pill"):
    st.session_state.shown_pill = choice
    st.session_state.place = choice

forecast = load_place(st.session_state.place)
if forecast.get("ok"):
    st.html(render_cards([forecast["card"]]))
else:
    st.error(forecast.get("error") or "That forecast could not be loaded.")

st.html(
    """
    <section class="ask">
      <h2>Ask the agent</h2>
      <p>Compare cities, ask if it will rain, or write in Urdu.</p>
    </section>
    """
)
st.caption("Forecasts come from Open-Meteo. No paid API key is needed.")

for message in st.session_state.messages:
    draw_message(message)

# Suggested questions only appear on an empty chat. After the first answer
# the user continues from the chat box, which stays on screen.
if not st.session_state.messages and pending is None:
    left, right = st.columns(2)
    for index, question in enumerate(PROMPTS):
        column = left if index % 2 == 0 else right
        if column.button(question, key=f"prompt-{index}", use_container_width=True):
            st.session_state["queued"] = question
            st.rerun()

if st.session_state.messages:
    if st.button("Clear chat", key="clear-chat"):
        st.session_state.messages = []
        st.rerun()

typed = st.chat_input("Ask about the weather...")
if typed:
    pending = typed
if pending:
    run_question(pending)
