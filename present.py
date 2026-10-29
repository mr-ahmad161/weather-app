"""HTML for the Sky page header, tool steps, and forecast cards.

Streamlit inserts these strings with st.html. Every user-facing value is
escaped. Theme colours are written as inline variables so each card can
follow its own weather without a separate stylesheet rule.
"""

from __future__ import annotations

import html

# Theme name -> (background, text, muted text, chip). Names match theme_for().
THEMES = {
    "clear-day": ("linear-gradient(160deg,#ffe8bf 0%,#ff8b3d 100%)", "#3a2012", "rgba(58,32,18,.72)", "rgba(255,255,255,.45)"),
    "clear-night": ("linear-gradient(165deg,#2c406e 0%,#12182b 100%)", "#f4f7ff", "rgba(244,247,255,.74)", "rgba(255,255,255,.1)"),
    "partly-day": ("linear-gradient(155deg,#ffd7a8 0%,#8ebbe8 100%)", "#1b2430", "rgba(27,36,48,.72)", "rgba(255,255,255,.42)"),
    "partly-night": ("linear-gradient(165deg,#31405f 0%,#171e30 100%)", "#eef3fb", "rgba(238,243,251,.75)", "rgba(255,255,255,.1)"),
    "cloud": ("linear-gradient(165deg,#6d8094 0%,#2c3846 100%)", "#f7f8fa", "rgba(247,248,250,.78)", "rgba(255,255,255,.12)"),
    "rain": ("linear-gradient(165deg,#1d679f 0%,#16324c 100%)", "#f3f8fc", "rgba(243,248,252,.78)", "rgba(255,255,255,.12)"),
    "snow": ("linear-gradient(180deg,#f7fbfe 0%,#d5e3ef 100%)", "#1c2a38", "rgba(28,42,56,.7)", "rgba(255,255,255,.72)"),
    "fog": ("linear-gradient(180deg,#e4eaee 0%,#a7b4be 100%)", "#1c242c", "rgba(28,36,44,.72)", "rgba(255,255,255,.5)"),
    "storm": ("linear-gradient(165deg,#53448f 0%,#241c3d 100%)", "#f7f4ff", "rgba(247,244,255,.78)", "rgba(255,255,255,.12)"),
}


def page_header() -> str:
    """Return the title block shown above the search box."""
    return """
    <header class="mast">
      <div>
        <p class="eyebrow">Weather agent</p>
        <h1>Sky</h1>
      </div>
    </header>
    """


def render_steps(steps: list[dict]) -> str:
    """Return one chip per tool call, marked failed when ok is false."""
    chips = []
    for index, step in enumerate(steps, start=1):
        failed = step.get("ok") is False
        kind = "bad" if failed else "ok"
        tool = html.escape(str(step.get("tool") or "tool"))
        detail = html.escape(str(step.get("detail") or ""))
        chips.append(f'<span class="step {kind}">{index}. {tool} · {detail}</span>')
    return '<div class="steps">' + "".join(chips) + "</div>"


def render_cards(cards: list[dict]) -> str:
    """Return a deck of forecast cards. An empty list renders nothing."""
    if not cards:
        return ""
    many = " many" if len(cards) > 1 else ""
    inner = "".join(_card(card) for card in cards)
    return f'<div class="deck{many}">{inner}</div>'


def _card(card: dict) -> str:
    """Render one place: condition, temperature, stats, hours, and 7 days."""
    background, ink, muted, chip = THEMES.get(card.get("theme") or "", THEMES["cloud"])
    place = card.get("place") or {}
    current = card.get("current") or {}
    daily = card.get("daily") or []
    today = daily[0] if daily else {}
    region = ", ".join(
        part
        for part in (place.get("admin1"), place.get("country"))
        if part and part != place.get("name")
    )
    wind = _plain(current.get("wind_speed"))
    direction = current.get("wind_direction") or ""
    wind_label = f"{wind} km/h" if wind != "—" else "—"
    if direction and wind != "—":
        wind_label = f"{wind} km/h {direction}"

    stats = [
        ("Humidity", f"{_plain(current.get('humidity'))}%"),
        ("Wind", wind_label),
        ("Rain", f"{_plain(today.get('precipitation_probability'))}%"),
        ("UV", _plain(today.get("uv_index"))),
    ]
    stat_html = "".join(
        f'<div class="stat"><small>{html.escape(label)}</small><b>{html.escape(value)}</b></div>'
        for label, value in stats
    )
    return (
        f'<article class="wx" style="background:{background};color:{ink};--muted:{muted};--chip:{chip};">'
        '<div class="wx-top">'
        "<div>"
        f'<div class="wx-name">{_esc(place.get("name"))}</div>'
        f'<div class="wx-region">{_esc(region)}</div>'
        "</div>"
        '<div class="wx-cond">'
        f'<span>{_esc(current.get("emoji"))}</span>'
        f'{_esc(current.get("condition"))}'
        "</div>"
        "</div>"
        f'<div class="wx-temp">{_plain(current.get("temperature"))}°</div>'
        '<div class="wx-feel">'
        f'Feels like {_plain(current.get("apparent_temperature"))}°'
        f' · Sunrise {_plain(today.get("sunrise"))}'
        f' · Sunset {_plain(today.get("sunset"))}'
        "</div>"
        f'<div class="stats">{stat_html}</div>'
        f'{_hours(card.get("hourly") or [])}'
        f'{_days(daily)}'
        "</article>"
    )


def _hours(hours: list[dict]) -> str:
    if not hours:
        return ""
    cells = []
    for hour in hours:
        cells.append(
            "<div class=\"hour\">"
            f'<div class="hour-label">{_esc(hour.get("label"))}</div>'
            f'<div class="hour-emoji">{_esc(hour.get("emoji"))}</div>'
            f'<b>{_plain(hour.get("temperature"))}°</b>'
            "</div>"
        )
    return '<div class="block-label">Next hours</div><div class="hours">' + "".join(cells) + "</div>"


def _days(days: list[dict]) -> str:
    if not days:
        return ""
    cells = []
    for day in days:
        cells.append(
            "<div class=\"day\">"
            f'<div class="wk">{_esc(day.get("weekday"))}</div>'
            f'<div class="em">{_esc(day.get("emoji"))}</div>'
            f'<div class="hi">{_plain(day.get("temp_max"))}°</div>'
            f'<div class="lo">{_plain(day.get("temp_min"))}°</div>'
            f'<div class="pop">{_plain(day.get("precipitation_probability"))}%</div>'
            "</div>"
        )
    return '<div class="block-label">Next 7 days</div><div class="days">' + "".join(cells) + "</div>"


def _esc(value) -> str:
    if value is None:
        return ""
    return html.escape(str(value))


def _plain(value) -> str:
    if value is None or value == "":
        return "—"
    return html.escape(str(value))
