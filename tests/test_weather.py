"""Offline tests for city parsing and forecast formatting.

These tests never call Open-Meteo. Place parsing is checked against the
questions the app suggests, and the weather helpers are checked with fixed
WMO codes and timestamps.
"""

import unittest

from agent import places_in_question
from weather import compass, condition_for, emoji_for, fetch_weather, hour_is_day, theme_for


class PlaceQuestionTests(unittest.TestCase):
    """The local agent must find the city in each suggested question."""
    def test_sample_questions_find_the_right_cities(self):
        self.assertEqual(places_in_question("Is it raining in Karachi?"), ["Karachi"])
        self.assertEqual(
            places_in_question("Compare the weather in Lahore and Islamabad."),
            ["Lahore", "Islamabad"],
        )
        self.assertEqual(places_in_question("Will it rain in Murree this week?"), ["Murree"])
        self.assertEqual(places_in_question("Lahore mein mausam kaisa hai?"), ["Lahore"])
        self.assertEqual(places_in_question("Tell me the current temperature in London."), ["London"])


class WeatherHelperTests(unittest.TestCase):
    """Condition labels, compass points, themes, and blank-input handling."""
    def test_conditions(self):
        self.assertEqual(condition_for(0), "Clear sky")
        self.assertEqual(condition_for(65), "Heavy rain")
        self.assertEqual(condition_for(95), "Thunderstorm")
        self.assertEqual(condition_for(None), "Unknown conditions")

    def test_compass_wraps_at_north(self):
        self.assertEqual(compass(0), "N")
        self.assertEqual(compass(45), "NE")
        self.assertEqual(compass(360), "N")

    def test_theme_and_emoji_follow_daylight(self):
        self.assertEqual(theme_for(0, True), "clear-day")
        self.assertEqual(theme_for(0, False), "clear-night")
        self.assertEqual(emoji_for(0, False), "🌙")
        self.assertEqual(theme_for(65, True), "rain")

    def test_hour_is_day_uses_sunrise_and_sunset(self):
        daily = [{"date": "2026-10-27", "sunrise": "06:14", "sunset": "17:18"}]
        self.assertTrue(hour_is_day("2026-10-27T15:00", daily))
        self.assertFalse(hour_is_day("2026-10-27T19:00", daily))
        self.assertTrue(hour_is_day("bad", daily, True))

    def test_blank_place_does_not_call_the_network(self):
        result = fetch_weather("   ")
        self.assertFalse(result["ok"])
        self.assertIn("city", result["error"].lower())


if __name__ == "__main__":
    unittest.main()
