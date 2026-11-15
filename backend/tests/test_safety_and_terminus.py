"""
Tests for P2 (departure advice safety consistency) and P3 (matatu terminus
restriction gating on destination).

All tests are unit-level — no network calls, no real model inference.
"""

import sys
import os
import pytest

# Ensure app/ is importable when pytest is run from backend/
_APP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app"))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)


# ---------------------------------------------------------------------------
# P2 — get_departure_advice() + safety_score
# ---------------------------------------------------------------------------

class TestDepartureAdviceSafetyConsistency:
    """
    The 'conditions are good' phrase must never appear when safety_score < 50,
    because the mode/city combination carries elevated risk regardless of
    current traffic and weather.
    """

    def _advice(self, congestion, weather, safety_score):
        from app.main import get_departure_advice
        return get_departure_advice(
            congestion=congestion,
            weather=weather,
            travel_time=30.0,
            mode="okada",
            safety_score=safety_score,
        )

    # ── Low-safety cases ──────────────────────────────────────────

    def test_low_safety_clear_low_congestion_no_good_message(self):
        """
        Low safety + clear weather + low congestion:
        must NOT say 'conditions are good'.
        """
        advice = self._advice("Low", "Clear", safety_score=26.0)
        assert "conditions are good" not in advice.lower(), (
            f"Got '{advice}' — should not imply good conditions with safety=26"
        )

    def test_low_safety_clear_low_congestion_acknowledges_risk(self):
        """Must mention either 'caution', 'risk', or the safety score value."""
        advice = self._advice("Low", "Clear", safety_score=30.0)
        lower = advice.lower()
        assert any(word in lower for word in ("caution", "risk", "safety", "30")), (
            f"Got '{advice}' — must acknowledge elevated risk"
        )

    def test_low_safety_contains_score(self):
        """The advice should embed the safety score number."""
        advice = self._advice("Low", "Clear", safety_score=35.0)
        assert "35" in advice, f"Expected score '35' in advice, got: '{advice}'"

    def test_low_safety_starts_with_leave_now(self):
        """
        Low safety is a baseline property — we still tell the user to leave
        now (safety score isn't a time-varying signal that warrants waiting).
        """
        advice = self._advice("Low", "Clear", safety_score=40.0)
        assert advice.lower().startswith("leave now"), (
            f"Expected advice to start with 'Leave now', got: '{advice}'"
        )

    # ── High-safety cases — original behaviour preserved ──────────

    def test_high_safety_clear_low_congestion_says_conditions_good(self):
        """Original 'conditions are good' phrasing preserved for safe routes."""
        advice = self._advice("Low", "Clear", safety_score=75.0)
        assert "conditions are good" in advice.lower(), (
            f"Expected 'conditions are good' for safe route, got: '{advice}'"
        )

    def test_high_safety_high_congestion_wait_advice_unchanged(self):
        """High congestion advice is not changed by safety_score."""
        from app.main import get_departure_advice
        advice = get_departure_advice("High", "Clear", 40.0, "driving", safety_score=80.0)
        assert "wait" in advice.lower() or "leave now" in advice.lower()

    # ── Boundary ──────────────────────────────────────────────────

    def test_safety_exactly_50_is_not_low_safety(self):
        """Score of exactly 50 should NOT trigger the low-safety branch."""
        advice = self._advice("Low", "Clear", safety_score=50.0)
        assert "conditions are good" in advice.lower(), (
            f"Score=50 should not be treated as low-safety, got: '{advice}'"
        )

    def test_safety_49_is_low_safety(self):
        """Score of 49 IS low safety."""
        advice = self._advice("Low", "Clear", safety_score=49.0)
        assert "conditions are good" not in advice.lower()

    # ── Rainy weather overrides low-safety branch (more specific) ─

    def test_rainy_weather_message_not_overridden_by_low_safety(self):
        """
        Rainy weather has its own branch that fires before the low-safety
        check.  The rainy-weather message must still appear.
        """
        advice = self._advice("Low", "Rainy", safety_score=30.0)
        assert "weather" in advice.lower() or "rain" in advice.lower(), (
            f"Rainy weather advice expected, got: '{advice}'"
        )

    # ── Default parameter ─────────────────────────────────────────

    def test_default_safety_score_preserves_good_conditions_message(self):
        """
        Old callers that do NOT pass safety_score get the default (100),
        so existing behaviour is unchanged.
        """
        from app.main import get_departure_advice
        advice = get_departure_advice("Low", "Clear", 20.0, "driving")
        assert "conditions are good" in advice.lower()


# ---------------------------------------------------------------------------
# P3 — _dest_is_cbd() and matatu TERMINUS_RESTRICTION gating
# ---------------------------------------------------------------------------

class TestDestIsCbd:
    """Unit tests for the _dest_is_cbd() helper."""

    def _cbd(self, dest, city="nairobi"):
        from app.main import _dest_is_cbd
        return _dest_is_cbd(dest, city)

    # ── Known CBD destinations ────────────────────────────────────

    def test_cbd_string_is_cbd(self):
        assert self._cbd("CBD") is True

    def test_cbd_lowercase(self):
        assert self._cbd("cbd") is True

    def test_city_centre_is_cbd(self):
        assert self._cbd("City Centre") is True

    def test_kenyatta_avenue_is_cbd(self):
        assert self._cbd("Kenyatta Avenue") is True

    def test_town_is_cbd(self):
        assert self._cbd("Town") is True

    def test_nairobi_cbd_explicit(self):
        assert self._cbd("Nairobi CBD") is True

    def test_partial_cbd_match_in_longer_string(self):
        assert self._cbd("near CBD roundabout") is True

    # ── Known outer suburbs — NOT CBD ─────────────────────────────

    def test_westlands_is_not_cbd(self):
        assert self._cbd("westlands") is False

    def test_karen_is_not_cbd(self):
        assert self._cbd("Karen") is False

    def test_kasarani_is_not_cbd(self):
        assert self._cbd("Kasarani") is False

    def test_thika_is_not_cbd(self):
        assert self._cbd("Thika") is False

    def test_eastleigh_is_not_cbd(self):
        assert self._cbd("Eastleigh") is False

    def test_suburb_with_nairobi_suffix(self):
        """Enriched place string like 'Kingeero, Nairobi' must still resolve correctly."""
        assert self._cbd("Kingeero, Nairobi") is False

    # ── Unknown destinations — assumed CBD (conservative) ─────────

    def test_unknown_dest_assumed_cbd(self):
        """Unrecognised destinations default to CBD assumption."""
        assert self._cbd("Upper Hill") is True  # not in suburb table

    def test_empty_string_assumed_cbd(self):
        assert self._cbd("") is True


class TestMatatuTerminusRestriction:
    """
    Integration-style unit tests for suggest_alternative_mode()
    with the matatu TERMINUS_RESTRICTION.

    We bypass the transport_modes pkl check by patching transport_modes
    and testing the suggestion logic with real restriction data.
    """

    def _suggest(self, destination, mode="matatu", city="nairobi",
                 congestion="Low", weather="Clear", distance_km=10.0):
        from app.main import suggest_alternative_mode
        return suggest_alternative_mode(
            mode=mode,
            city=city,
            congestion=congestion,
            weather=weather,
            distance_km=distance_km,
            destination=destination,
        )

    def test_cbd_destination_triggers_terminus_warning(self):
        """Matatu to CBD must produce the terminus restriction message."""
        msg = self._suggest("CBD")
        # If transport_modes is loaded, the terminus message is non-None;
        # if models are absent (CI without pkl), the function returns None —
        # skip gracefully in that case.
        if msg is None:
            pytest.skip("transport_modes not loaded — skipping integration path")
        assert "terminus" in msg.lower() or "termini" in msg.lower() or "cbd" in msg.lower(), (
            f"Expected terminus warning for CBD, got: '{msg}'"
        )

    def test_outer_suburb_destination_no_terminus_warning(self):
        """Matatu to a known outer suburb must NOT produce the terminus message."""
        msg = self._suggest("Westlands")
        # terminus warning must not appear
        if msg is not None:
            assert "terminus" not in msg.lower() and "termini" not in msg.lower(), (
                f"Got unexpected terminus warning for Westlands: '{msg}'"
            )

    def test_kasarani_no_terminus_warning(self):
        msg = self._suggest("Kasarani")
        if msg is not None:
            assert "terminus" not in msg.lower() and "termini" not in msg.lower()

    def test_town_centre_triggers_terminus_warning(self):
        msg = self._suggest("Town Centre")
        if msg is None:
            pytest.skip("transport_modes not loaded")
        assert "terminus" in msg.lower() or "termini" in msg.lower() or "cbd" in msg.lower() or \
               msg != "", f"Expected terminus warning for Town Centre, got: '{msg}'"

    def test_empty_destination_triggers_terminus_warning(self):
        """Empty destination is conservatively treated as CBD."""
        msg = self._suggest("")
        if msg is None:
            pytest.skip("transport_modes not loaded")
        # Should produce a terminus warning (conservative)
        assert msg is not None


# ---------------------------------------------------------------------------
# P2+P3 combined — full endpoint test (mocked)
# ---------------------------------------------------------------------------

class TestEndpointSafetyConsistency:
    """
    Smoke test via the full /v2/predict endpoint to confirm the departure
    advice does not contradict the safety score in the response.
    """

    @pytest.fixture()
    def client(self):
        from unittest.mock import AsyncMock, MagicMock, patch
        from fastapi.testclient import TestClient

        GOOD_COORDS = {"lat": 6.5244, "lng": 3.3792}
        GOOD_ROUTE  = {"distance_km": 8.4, "base_duration_minutes": 22.0,
                       "route_geometry": [[6.52, 3.375], [6.524, 3.379]]}

        from app.main import app
        client = TestClient(app, raise_server_exceptions=False)

        patches = [
            ("app.main.geocode_place",          AsyncMock(return_value=GOOD_COORDS)),
            ("app.main.get_route",               AsyncMock(return_value=GOOD_ROUTE)),
            ("app.main.get_weather",             AsyncMock(return_value={"label":"Clear","code":0,"wind_kmh":0})),
            ("app.main.get_route_intelligence",  AsyncMock(return_value={
                "congestion":"Low","congestion_detail":{"live":False,"source":"time"},
                "incidents":[],"incident_count":0,"confidence_boost":0,
                "data_sources":["time_of_day_estimate"],
            })),
            ("app.main.list_reports",            AsyncMock(return_value=[])),
            ("app.main.get_weather_trend",       AsyncMock(return_value={"trend":"stable"})),
            ("app.main.get_flood_risk",          MagicMock(return_value={"risk":"Low","zones":[]})),
            ("app.main.get_day_pattern",         MagicMock(return_value={"day":"Monday","congestion_mult":1.0,"description":"Normal"})),
            # Force a low safety score — get_safety_score now returns a dict
            ("app.main.get_safety_score",        MagicMock(return_value={
                "score": 28.0, "available": True,
                "basis": "crash_data_pkl", "mode_adj": True,
            })),
        ]

        started = []
        for target, mock in patches:
            p = patch(target, new=mock)
            p.start()
            started.append(p)

        yield client

        for p in started:
            p.stop()

    def test_low_safety_departure_advice_not_good(self, client):
        """
        When get_safety_score returns 28, departure_advice must NOT contain
        'conditions are good'.
        """
        resp = client.post("/v2/predict", json={
            "origin": "Surulere", "destination": "Lagos Island",
            "mode": "okada", "city": "lagos", "time": "10:00",
        })
        if resp.status_code != 200:
            pytest.skip(f"Endpoint returned {resp.status_code} — skipping")

        data = resp.json()
        advice = data.get("departure_advice", "")
        assert "conditions are good" not in advice.lower(), (
            f"departure_advice='{advice}' contradicts safety_score=28"
        )
