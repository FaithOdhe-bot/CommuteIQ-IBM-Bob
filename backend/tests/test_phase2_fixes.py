"""
test_phase2_fixes.py — Phase 2 regression tests (Task D)

Covers:
  A1 — Travel-time sanity guard (negative / NaN / zero / too-large rejection)
  A5 — Road quality availability flag for all 13 cities
  A6 — RandomForest named-DataFrame fix (no UserWarning)
  A3/A4 — Safety score provenance fields (all 13 cities)
  B  — live_intelligence congestion_note present in v2/predict response
  C  — API_BASE meta tag present in frontend/index.html

None of these tests modify pkl files or retrain models.
"""

import math
import pathlib
import sys
import warnings

import pytest

# ---------------------------------------------------------------------------
# Ensure app module is importable when pytest is run from backend/
# ---------------------------------------------------------------------------
APP_DIR = pathlib.Path(__file__).parent.parent / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))


# ===========================================================================
# A1 — Travel-time sanity guard
# ===========================================================================

class TestTravelTimeSanityGuard:
    """_travel_time_is_valid() must reject non-finite, negative,
    zero, and implausibly large values and accept valid ones."""

    def _load(self):
        from main import _travel_time_is_valid, _MIN_TRAVEL_TIME_MIN, _MAX_TRAVEL_TIME_MIN
        return _travel_time_is_valid, _MIN_TRAVEL_TIME_MIN, _MAX_TRAVEL_TIME_MIN

    def test_negative_travel_time_is_invalid(self):
        f, _, _ = self._load()
        assert f(-21.6) is False, "Negative travel time must be rejected"

    def test_zero_travel_time_is_invalid(self):
        f, _, _ = self._load()
        assert f(0.0) is False, "Zero travel time must be rejected"

    def test_nan_is_invalid(self):
        f, _, _ = self._load()
        assert f(float("nan")) is False, "NaN travel time must be rejected"

    def test_positive_infinity_is_invalid(self):
        f, _, _ = self._load()
        assert f(float("inf")) is False, "+Inf travel time must be rejected"

    def test_negative_infinity_is_invalid(self):
        f, _, _ = self._load()
        assert f(float("-inf")) is False, "-Inf travel time must be rejected"

    def test_too_large_is_invalid(self):
        f, _, _max = self._load()
        assert f(_max + 1) is False, "Value above _MAX_TRAVEL_TIME_MIN must be rejected"

    def test_minimum_bound_is_valid(self):
        f, _min, _ = self._load()
        assert f(_min) is True, f"Exactly {_min} min should be valid"

    def test_maximum_bound_is_valid(self):
        f, _, _max = self._load()
        assert f(_max) is True, f"Exactly {_max} min should be valid"

    def test_typical_urban_time_is_valid(self):
        f, _, _ = self._load()
        assert f(35.5) is True, "35.5 min is a plausible urban commute"

    def test_one_minute_exactly_is_valid(self):
        f, _, _ = self._load()
        assert f(1.0) is True

    def test_min_constant_is_one(self):
        _, _min, _ = self._load()
        assert _min == 1.0, "_MIN_TRAVEL_TIME_MIN should be 1.0 min"

    def test_max_constant_is_600(self):
        _, _, _max = self._load()
        assert _max == 600.0, "_MAX_TRAVEL_TIME_MIN should be 600.0 min (10 hours)"

    def test_predict_travel_time_formula_fallback_for_short_trip(self):
        """predict_travel_time must use formula (not XGBoost) for sub-50 km trips."""
        from main import predict_travel_time
        # 10 km trip — well below 50 km XGBoost threshold
        t = predict_travel_time(10.0, "Low", "Clear", 2, "driving", "lagos", 50.0)
        assert isinstance(t, float), "Result should be a float"
        assert _travel_time_is_valid_helper(t), f"Formula result {t} should be plausible"

    def test_predict_travel_time_never_returns_negative(self):
        """Formula fallback must never produce a negative result for a positive distance."""
        from main import predict_travel_time
        for dist in [1.0, 5.0, 12.0, 25.0, 45.0]:
            t = predict_travel_time(dist, "High", "Rainy", 2, "driving", "nairobi", 22.3)
            assert t > 0, f"predict_travel_time({dist}) returned {t} — must be positive"


def _travel_time_is_valid_helper(t):
    return isinstance(t, (int, float)) and math.isfinite(t) and 1.0 <= t <= 600.0


# ===========================================================================
# A5 — Road quality availability for all 13 cities
# ===========================================================================

ALL_CITIES = [
    "lagos", "abuja", "kano", "ibadan", "port harcourt", "enugu",
    "nairobi", "mombasa", "kisumu", "nakuru", "eldoret",
    "asaba", "benin city",
]

# Only these 4 cities have real data in road_quality.pkl city_avg.
ROAD_QUALITY_DATA_CITIES = {"lagos", "kano", "abuja", "nairobi"}


class TestRoadQualityAvailability:
    """get_road_quality_score() must return a dict with 'available', 'score',
    and 'source' for every supported city. Cities without data should
    explicitly report available=False, not silently return a fake 50.0."""

    def _fn(self):
        from main import get_road_quality_score
        return get_road_quality_score

    def test_returns_dict(self):
        fn = self._fn()
        result = fn("lagos")
        assert isinstance(result, dict), "Must return a dict"

    def test_dict_has_required_keys(self):
        fn = self._fn()
        result = fn("lagos")
        for key in ("score", "available", "source"):
            assert key in result, f"Key '{key}' missing from road quality result"

    @pytest.mark.parametrize("city", ALL_CITIES)
    def test_score_is_float_in_range(self, city):
        fn = self._fn()
        result = fn(city)
        assert isinstance(result["score"], float), f"{city}: score must be float"
        assert 0.0 <= result["score"] <= 100.0, f"{city}: score {result['score']} out of range"

    @pytest.mark.parametrize("city", list(ROAD_QUALITY_DATA_CITIES))
    def test_data_cities_have_available_true(self, city):
        fn = self._fn()
        result = fn(city)
        assert result["available"] is True, (
            f"{city} has road quality data in pkl; available should be True"
        )

    @pytest.mark.parametrize("city", [c for c in ALL_CITIES if c not in ROAD_QUALITY_DATA_CITIES])
    def test_no_data_cities_have_available_false(self, city):
        fn = self._fn()
        result = fn(city)
        assert result["available"] is False, (
            f"{city} has NO road quality data; available should be False, got True"
        )

    @pytest.mark.parametrize("city", [c for c in ALL_CITIES if c not in ROAD_QUALITY_DATA_CITIES])
    def test_no_data_cities_return_unavailable_fallback_source(self, city):
        fn = self._fn()
        result = fn(city)
        assert result["source"] == "unavailable_fallback", (
            f"{city}: source should be 'unavailable_fallback', got {result['source']!r}"
        )

    @pytest.mark.parametrize("city", list(ROAD_QUALITY_DATA_CITIES))
    def test_data_cities_source_is_city_avg_pkl(self, city):
        fn = self._fn()
        result = fn(city)
        assert result["source"] == "city_avg_pkl", (
            f"{city}: source should be 'city_avg_pkl', got {result['source']!r}"
        )

    def test_unavailable_score_is_exactly_50(self):
        """The neutral fallback sentinel must be exactly 50.0 (midpoint — no directional bias)."""
        from main import get_road_quality_score, ROAD_QUALITY_UNAVAILABLE
        result = get_road_quality_score("enugu")  # confirmed no-data city
        assert result["score"] == ROAD_QUALITY_UNAVAILABLE == 50.0


# ===========================================================================
# A6 — RandomForest named-DataFrame fix (no UserWarning)
# ===========================================================================

class TestRandomForestNamedDataFrame:
    """get_commute_quality() must not emit sklearn's
    'X does not have valid feature names' UserWarning."""

    def test_no_feature_names_warning(self):
        from main import get_commute_quality
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            get_commute_quality(
                congestion="High", weather="Rainy",
                community_reports=0, safety_score=65.0,
                mode="driving", city="lagos",
                distance_km=12.0, rq_score=27.1,
            )
        feature_warnings = [
            w for w in caught
            if "feature names" in str(w.message).lower()
            or "feature_names_in_" in str(w.message)
        ]
        assert len(feature_warnings) == 0, (
            f"Unexpected sklearn feature-name warning(s): "
            + ", ".join(str(w.message) for w in feature_warnings)
        )

    def test_returns_dict_with_required_keys(self):
        from main import get_commute_quality
        result = get_commute_quality(
            congestion="Low", weather="Clear",
            community_reports=0, safety_score=70.0,
            mode="driving", city="nairobi",
            distance_km=8.0, rq_score=22.3,
        )
        for key in ("label", "emoji", "score"):
            assert key in result, f"Key '{key}' missing from commute quality result"

    def test_quality_label_is_valid_string(self):
        from main import get_commute_quality
        result = get_commute_quality(
            congestion="Medium", weather="Clear",
            community_reports=1, safety_score=55.0,
            mode="matatu", city="nairobi",
            distance_km=5.0, rq_score=22.3,
        )
        assert result["label"] in ("Good", "Moderate", "Poor"), (
            f"Unexpected quality label: {result['label']!r}"
        )

    def test_quality_score_is_int_in_range(self):
        from main import get_commute_quality
        result = get_commute_quality(
            congestion="Low", weather="Clear",
            community_reports=0, safety_score=80.0,
            mode="driving", city="abuja",
            distance_km=10.0, rq_score=27.1,
        )
        assert isinstance(result["score"], int), "quality score must be int"
        assert result["score"] in (30, 60, 90), f"Unexpected score: {result['score']}"


# ===========================================================================
# A3/A4 — Safety score provenance fields
# ===========================================================================

HARDCODED_CITIES = {"nairobi", "lagos"}


class TestSafetyScoreProvenance:
    """get_safety_score() must return all 7 provenance fields for every city."""

    def _fn(self):
        from main import get_safety_score
        return get_safety_score

    REQUIRED_KEYS = {
        "score", "available", "basis", "mode_adj",
        "is_hardcoded_override", "geographic_granularity",
        "is_historical_indicator",
    }

    @pytest.mark.parametrize("city", ALL_CITIES)
    def test_all_provenance_keys_present(self, city):
        fn = self._fn()
        result = fn(city, "driving")
        for key in self.REQUIRED_KEYS:
            assert key in result, (
                f"{city}: provenance key '{key}' missing from get_safety_score()"
            )

    @pytest.mark.parametrize("city", ALL_CITIES)
    def test_is_historical_indicator_always_true(self, city):
        fn = self._fn()
        result = fn(city, "driving")
        assert result["is_historical_indicator"] is True, (
            f"{city}: is_historical_indicator must always be True"
        )

    @pytest.mark.parametrize("city", list(HARDCODED_CITIES))
    def test_hardcoded_cities_flagged(self, city):
        fn = self._fn()
        result = fn(city, "driving")
        assert result["is_hardcoded_override"] is True, (
            f"{city}: is_hardcoded_override should be True (manually set estimate in train_models.py)"
        )

    @pytest.mark.parametrize("city", [c for c in ALL_CITIES if c not in HARDCODED_CITIES])
    def test_non_hardcoded_cities_not_flagged(self, city):
        fn = self._fn()
        result = fn(city, "driving")
        assert result["is_hardcoded_override"] is False, (
            f"{city}: is_hardcoded_override should be False (formula-derived from crash CSV)"
        )

    @pytest.mark.parametrize("city", list(HARDCODED_CITIES))
    def test_hardcoded_cities_basis_is_estimate(self, city):
        fn = self._fn()
        result = fn(city, "driving")
        assert result["basis"] == "hardcoded_city_estimate", (
            f"{city}: basis should be 'hardcoded_city_estimate'"
        )

    @pytest.mark.parametrize("city", [c for c in ALL_CITIES if c not in HARDCODED_CITIES])
    def test_formula_cities_basis_is_crash_data(self, city):
        fn = self._fn()
        result = fn(city, "driving")
        assert result["basis"] == "crash_data_formula_pkl", (
            f"{city}: basis should be 'crash_data_formula_pkl'"
        )

    @pytest.mark.parametrize("city", ALL_CITIES)
    def test_geographic_granularity_is_known_string(self, city):
        fn = self._fn()
        result = fn(city, "driving")
        assert result["geographic_granularity"] in ("state", "county", "city_estimate"), (
            f"{city}: unexpected granularity value {result['geographic_granularity']!r}"
        )

    def test_nairobi_granularity_is_city_estimate(self):
        fn = self._fn()
        result = fn("nairobi", "driving")
        assert result["geographic_granularity"] == "city_estimate"

    def test_lagos_granularity_is_city_estimate(self):
        fn = self._fn()
        result = fn("lagos", "driving")
        assert result["geographic_granularity"] == "city_estimate"

    def test_abuja_granularity_is_state(self):
        fn = self._fn()
        result = fn("abuja", "driving")
        assert result["geographic_granularity"] == "state"

    def test_mombasa_granularity_is_county(self):
        fn = self._fn()
        result = fn("mombasa", "driving")
        assert result["geographic_granularity"] == "county"


# ===========================================================================
# B — live_intelligence congestion_note in /v2/predict
# ===========================================================================

@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from main import app
    return TestClient(app)


SAMPLE_V2_PAYLOAD = {
    "origin":      "Westlands",
    "destination": "CBD",
    "mode":        "matatu",
    "city":        "nairobi",
    "time":        "08:00",
}


class TestLiveIntelligenceCongestionNote:
    """When TomTom is not configured, /v2/predict must explicitly declare
    that congestion is an estimate via congestion_note."""

    def test_live_intelligence_field_present(self, client):
        resp = client.post("/v2/predict", json=SAMPLE_V2_PAYLOAD)
        # Accept 200 (normal) or 503 (OSRM unreachable in test env)
        if resp.status_code == 503:
            pytest.skip("OSRM routing unavailable in test environment")
        assert resp.status_code == 200
        body = resp.json()
        assert "live_intelligence" in body, "'live_intelligence' key missing from /v2/predict response"

    def test_congestion_note_present(self, client):
        resp = client.post("/v2/predict", json=SAMPLE_V2_PAYLOAD)
        if resp.status_code == 503:
            pytest.skip("OSRM routing unavailable in test environment")
        body = resp.json()
        intel = body.get("live_intelligence", {})
        assert "congestion_note" in intel, (
            "'congestion_note' missing from live_intelligence block"
        )

    def test_congestion_method_present(self, client):
        resp = client.post("/v2/predict", json=SAMPLE_V2_PAYLOAD)
        if resp.status_code == 503:
            pytest.skip("OSRM routing unavailable in test environment")
        body = resp.json()
        intel = body.get("live_intelligence", {})
        assert "congestion_method" in intel, (
            "'congestion_method' missing from live_intelligence block"
        )

    def test_no_tomtom_gives_heuristic_method(self, client):
        """Without TOMTOM_API_KEY set, method must be 'time_of_day_heuristic'."""
        import os
        if os.getenv("TOMTOM_API_KEY"):
            pytest.skip("TOMTOM_API_KEY is set — TomTom is active, skip no-TomTom assertion")
        resp = client.post("/v2/predict", json=SAMPLE_V2_PAYLOAD)
        if resp.status_code == 503:
            pytest.skip("OSRM routing unavailable in test environment")
        body = resp.json()
        method = body.get("live_intelligence", {}).get("congestion_method")
        assert method == "time_of_day_heuristic", (
            f"Expected 'time_of_day_heuristic' without TomTom, got {method!r}"
        )

    def test_no_tomtom_congestion_note_is_informative_string(self, client):
        import os
        if os.getenv("TOMTOM_API_KEY"):
            pytest.skip("TOMTOM_API_KEY is set")
        resp = client.post("/v2/predict", json=SAMPLE_V2_PAYLOAD)
        if resp.status_code == 503:
            pytest.skip("OSRM routing unavailable in test environment")
        body = resp.json()
        note = body.get("live_intelligence", {}).get("congestion_note")
        assert note is not None, "congestion_note should not be None when TomTom is inactive"
        assert "estimated" in note.lower() or "time-of-day" in note.lower(), (
            f"congestion_note should mention estimation/time-of-day, got: {note!r}"
        )


# ===========================================================================
# B — road_quality_available + safety provenance in /v2/predict response
# ===========================================================================

class TestV2PredictProvenanceFields:
    """The /v2/predict endpoint must return all Phase 2 provenance fields."""

    def test_road_quality_available_field_present(self, client):
        resp = client.post("/v2/predict", json=SAMPLE_V2_PAYLOAD)
        if resp.status_code == 503:
            pytest.skip("OSRM routing unavailable in test environment")
        body = resp.json()
        assert "road_quality_available" in body

    def test_road_quality_source_field_present(self, client):
        resp = client.post("/v2/predict", json=SAMPLE_V2_PAYLOAD)
        if resp.status_code == 503:
            pytest.skip("OSRM routing unavailable in test environment")
        body = resp.json()
        assert "road_quality_source" in body

    def test_safety_score_is_hardcoded_field_present(self, client):
        resp = client.post("/v2/predict", json=SAMPLE_V2_PAYLOAD)
        if resp.status_code == 503:
            pytest.skip("OSRM routing unavailable in test environment")
        body = resp.json()
        assert "safety_score_is_hardcoded" in body

    def test_safety_score_granularity_field_present(self, client):
        resp = client.post("/v2/predict", json=SAMPLE_V2_PAYLOAD)
        if resp.status_code == 503:
            pytest.skip("OSRM routing unavailable in test environment")
        body = resp.json()
        assert "safety_score_granularity" in body

    def test_safety_score_is_historical_field_present(self, client):
        resp = client.post("/v2/predict", json=SAMPLE_V2_PAYLOAD)
        if resp.status_code == 503:
            pytest.skip("OSRM routing unavailable in test environment")
        body = resp.json()
        assert "safety_score_is_historical" in body
        assert body["safety_score_is_historical"] is True

    def test_nairobi_safety_is_flagged_as_hardcoded(self, client):
        resp = client.post("/v2/predict", json=SAMPLE_V2_PAYLOAD)  # city=nairobi
        if resp.status_code == 503:
            pytest.skip("OSRM routing unavailable in test environment")
        body = resp.json()
        assert body.get("safety_score_is_hardcoded") is True, (
            "Nairobi safety score is a hardcoded estimate — must be flagged"
        )

    def test_nairobi_road_quality_available_is_true(self, client):
        resp = client.post("/v2/predict", json=SAMPLE_V2_PAYLOAD)  # city=nairobi
        if resp.status_code == 503:
            pytest.skip("OSRM routing unavailable in test environment")
        body = resp.json()
        # Nairobi is in _ROAD_QUALITY_CITY_AVG_CITIES
        assert body.get("road_quality_available") is True, (
            "Nairobi has city_avg road quality data — available should be True"
        )

    def test_enugu_road_quality_available_is_false(self, client):
        payload = {**SAMPLE_V2_PAYLOAD, "city": "enugu",
                   "mode": "driving", "origin": "GRA", "destination": "New Haven"}
        resp = client.post("/v2/predict", json=payload)
        if resp.status_code == 503:
            pytest.skip("OSRM routing unavailable in test environment")
        body = resp.json()
        assert body.get("road_quality_available") is False, (
            "Enugu has no road quality data — available should be False"
        )


# ===========================================================================
# C — API_BASE meta tag in frontend/index.html
# ===========================================================================

class TestFrontendApiBase:
    """index.html must contain the configurable API_BASE meta tag added in Phase 2."""

    def _html(self):
        html_path = (
            pathlib.Path(__file__).parent.parent.parent
            / "frontend" / "index.html"
        )
        assert html_path.exists(), f"frontend/index.html not found at {html_path}"
        return html_path.read_text(encoding="utf-8")

    def test_meta_commuteiq_api_base_present(self):
        html = self._html()
        assert 'name="commuteiq-api-base"' in html, (
            "meta[name='commuteiq-api-base'] missing — API_BASE is not configurable"
        )

    def test_meta_has_content_attribute(self):
        html = self._html()
        # The tag should have a content attribute (may be empty for default)
        assert 'commuteiq-api-base' in html and 'content=' in html

    def test_no_hardcoded_localhost_const(self):
        """The old `const API_BASE = "http://localhost:8000"` must not exist
        without the new IIFE wrapping it."""
        html = self._html()
        # The hardcoded assignment pattern is gone; the IIFE replaces it.
        # Allow the string to appear inside a comment or inside the IIFE fallback,
        # but it should NOT appear as a bare top-level `const API_BASE = "http://localhost:8000"`.
        import re
        bare_assignment = re.search(
            r'^\s*const\s+API_BASE\s*=\s*["\']http://localhost:8000["\']',
            html,
            re.MULTILINE,
        )
        assert bare_assignment is None, (
            "Hardcoded `const API_BASE = 'http://localhost:8000'` still present — "
            "should be replaced with the meta-tag IIFE"
        )

    def test_window_location_origin_fallback_present(self):
        """The IIFE should fall back to window.location.origin for same-origin serving."""
        html = self._html()
        assert "window.location.origin" in html, (
            "window.location.origin fallback missing from API_BASE IIFE"
        )
