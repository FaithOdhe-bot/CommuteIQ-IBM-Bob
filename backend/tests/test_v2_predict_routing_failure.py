"""
Regression tests for POST /v2/predict routing-failure handling.

Prior to the fix, any exception raised inside the routing try-block left
three variables unbound:
  - route_geometry   (NameError at response construction, line ~1438)
  - origin_coords    (NameError inside walk-to-transit branch, line ~1378)
  - dest_coords      (accessed via fragile `'dest_coords' in dir()` guard)

The fixed endpoint raises HTTP 503 with a descriptive message instead of
crashing with HTTP 500 / UnboundLocalError.

All external I/O is mocked — no real network calls are made.
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Shared mock payloads used across tests
# ---------------------------------------------------------------------------

GOOD_COORDS  = {"lat": 6.5244, "lng": 3.3792}
GOOD_ROUTE   = {
    "distance_km": 8.4,
    "base_duration_minutes": 22.0,
    "route_geometry": [
        [6.520, 3.375],
        [6.522, 3.377],
        [6.524, 3.379],
    ],
}

# Minimal weather / intelligence / storage mocks so the happy-path reaches
# the return statement without touching real services.
WEATHER_OK        = {"label": "Clear", "code": 0, "wind_kmh": 5}
INTEL_OK          = {
    "congestion": "Low",
    "congestion_detail": {"source": "time_of_day_estimate", "live": False},
    "incidents": [],
    "incident_count": 0,
    "confidence_boost": 0,
    "data_sources": ["time_of_day_estimate"],
}
WEATHER_TREND_OK  = {"trend": "stable", "rain_in_30min": False}
FLOOD_RISK_OK     = {"risk": "Low", "zones": []}
DAY_PATTERN_OK    = {"day": "Monday", "congestion_mult": 1.0, "description": "Normal"}

VALID_REQUEST = {
    "origin":      "Westlands",
    "destination": "CBD",
    "mode":        "driving",
    "city":        "nairobi",
    "time":        "08:00",
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def client():
    """
    Import the FastAPI app only after the fixture starts so that module-level
    model loading doesn't interfere.  All external calls are patched at the
    app.main module level — that is where the names are actually looked up.
    """
    import sys
    import os

    # Make sure backend/app is importable
    app_dir = os.path.join(os.path.dirname(__file__), "..", "app")
    if app_dir not in sys.path:
        sys.path.insert(0, os.path.abspath(app_dir))

    from app.main import app
    return TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# Helper — builds the full patch stack used by the happy-path test
# ---------------------------------------------------------------------------

def _all_external_mocks(
    geocode_side_effect=None,
    route_side_effect=None,
    geocode_return=None,
    route_return=None,
):
    """
    Returns a list of (target, kwargs) pairs ready for use with
    unittest.mock.patch.  Callers supply overrides for the geocode /
    routing calls; everything else is silenced with sensible defaults.
    """
    geo_mock   = AsyncMock(return_value=geocode_return or GOOD_COORDS,
                           side_effect=geocode_side_effect)
    route_mock = AsyncMock(return_value=route_return or GOOD_ROUTE,
                           side_effect=route_side_effect)

    return [
        ("app.main.geocode_place",        {"new": geo_mock}),
        ("app.main.get_route",            {"new": route_mock}),
        ("app.main.get_weather",          {"new": AsyncMock(return_value=WEATHER_OK)}),
        ("app.main.get_route_intelligence", {"new": AsyncMock(return_value=INTEL_OK)}),
        ("app.main.list_reports",         {"new": AsyncMock(return_value=[])}),
        ("app.main.get_weather_trend",    {"new": AsyncMock(return_value=WEATHER_TREND_OK)}),
        ("app.main.get_flood_risk",       {"new": MagicMock(return_value=FLOOD_RISK_OK)}),
        ("app.main.get_day_pattern",      {"new": MagicMock(return_value=DAY_PATTERN_OK)}),
    ]


# ---------------------------------------------------------------------------
# REGRESSION — geocode raises → was NameError (route_geometry / origin_coords)
# ---------------------------------------------------------------------------

class TestV2PredictRoutingFailure:

    def _post(self, client, overrides=None):
        body = {**VALID_REQUEST, **(overrides or {})}
        patches = _all_external_mocks(
            geocode_side_effect=Exception("Nominatim unreachable"),
        )
        ctx = [patch(t, **kw) for t, kw in patches]
        for p in ctx:
            p.start()
        try:
            return client.post("/v2/predict", json=body)
        finally:
            for p in ctx:
                p.stop()

    def test_geocode_failure_returns_503_not_500(self, client):
        """
        Before the fix this raised UnboundLocalError → HTTP 500.
        After the fix it must return HTTP 503.
        """
        response = self._post(client)
        assert response.status_code == 503, (
            f"Expected 503, got {response.status_code}. "
            f"Body: {response.text[:300]}"
        )

    def test_503_body_contains_descriptive_message(self, client):
        """The error detail must name both places and the city."""
        response = self._post(client)
        detail = response.json().get("detail", "")
        assert "Westlands" in detail
        assert "CBD" in detail
        assert "Nairobi" in detail   # city is title-cased in the message

    def test_no_unbound_variable_error_in_response(self, client):
        """Ensure the response body is not a raw Python traceback."""
        response = self._post(client)
        body = response.text
        assert "NameError" not in body
        assert "UnboundLocalError" not in body
        assert "route_geometry" not in body   # not leaked in error text


# ---------------------------------------------------------------------------
# REGRESSION — OSRM raises after geocoding succeeds
# ---------------------------------------------------------------------------

class TestV2PredictOsrmFailure:

    def _post(self, client):
        patches = _all_external_mocks(
            geocode_return=GOOD_COORDS,
            route_side_effect=Exception("OSRM connection refused"),
        )
        ctx = [patch(t, **kw) for t, kw in patches]
        for p in ctx:
            p.start()
        try:
            return client.post("/v2/predict", json=VALID_REQUEST)
        finally:
            for p in ctx:
                p.stop()

    def test_osrm_failure_returns_503(self, client):
        """OSRM down → HTTP 503 (not 500 / NameError on route_geometry)."""
        response = self._post(client)
        assert response.status_code == 503

    def test_osrm_503_detail_mentions_routing_service(self, client):
        detail = self._post(client).json().get("detail", "")
        assert "routing" in detail.lower() or "route" in detail.lower()


# ---------------------------------------------------------------------------
# HAPPY PATH — successful routing must still return 200 with full schema
# ---------------------------------------------------------------------------

class TestV2PredictHappyPath:

    def _post(self, client, body=None):
        patches = _all_external_mocks()
        ctx = [patch(t, **kw) for t, kw in patches]
        for p in ctx:
            p.start()
        try:
            return client.post("/v2/predict", json=body or VALID_REQUEST)
        finally:
            for p in ctx:
                p.stop()

    def test_returns_200(self, client):
        assert self._post(client).status_code == 200

    def test_distance_km_is_from_osrm(self, client):
        """distance_km must reflect the mocked OSRM value, not any hardcoded fallback."""
        data = self._post(client).json()
        assert data["distance_km"] == pytest.approx(GOOD_ROUTE["distance_km"], abs=0.1)

    def test_route_geometry_present(self, client):
        data = self._post(client).json()
        assert data["route_geometry"] is not None
        assert len(data["route_geometry"]) == len(GOOD_ROUTE["route_geometry"])

    def test_required_fields_present(self, client):
        data = self._post(client).json()
        for field in [
            "travel_time_min", "commute_quality", "safety_score",
            "weather", "congestion", "departure_advice", "ai_explanation",
        ]:
            assert field in data, f"Missing field: {field}"

    def test_city_validation_still_works(self, client):
        """Invalid city must still return 400."""
        patches = _all_external_mocks()
        ctx = [patch(t, **kw) for t, kw in patches]
        for p in ctx:
            p.start()
        try:
            resp = client.post("/v2/predict", json={**VALID_REQUEST, "city": "atlantis"})
        finally:
            for p in ctx:
                p.stop()
        assert resp.status_code == 400
