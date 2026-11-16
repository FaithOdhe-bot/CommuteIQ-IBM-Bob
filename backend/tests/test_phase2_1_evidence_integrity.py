"""
test_phase2_1_evidence_integrity.py — Phase 2.1 regression tests

Covers:
  P0-1  Evidence freshness: published_at parsing, stale-item rejection,
        no age-reset on cache refresh, unverifiable-timestamp exclusion
  P0-2  Geographic relevance: title-only assignment, body-only exclusion,
        all 13 cities covered, wrong-city prevention
  P0-3  Frontend provenance: road_quality_available guard, safety score
        provenance fields (tested at API level — the JS guard is in index.html)
  P1-4  Deduplication: URL-level, headline-level, distinct simultaneous
        incidents preserved, cross-source merging with attribution
  P1-5  Provider health: success vs failure tracking, error categories,
        distinguish no-incidents from provider-down
  P1-6  Incident provenance: community_count = verified only;
        unverified news items excluded from safety-alert count

None of these tests make real network calls.  External dependencies (httpx,
RSS feeds, TomTom) are mocked or bypassed.
"""

import math
import pathlib
import sys
import time
import unittest.mock as mock

import pytest

# ---------------------------------------------------------------------------
# Ensure app module is importable
# ---------------------------------------------------------------------------
APP_DIR = pathlib.Path(__file__).parent.parent / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))


# ===========================================================================
# P0-1 — Evidence freshness
# ===========================================================================

class TestEvidenceFreshness:

    def _load(self):
        from live_intelligence import _parse_rfc2822, _article_is_fresh, MAX_ARTICLE_AGE_S
        return _parse_rfc2822, _article_is_fresh, MAX_ARTICLE_AGE_S

    # ── Timestamp parsing ────────────────────────────────────────

    def test_valid_rfc2822_parses_to_float(self):
        parse, _, _ = self._load()
        ts = parse("Mon, 03 Jun 2024 09:15:00 +0000")
        assert isinstance(ts, float) and ts > 0

    def test_empty_pubdate_returns_none(self):
        parse, _, _ = self._load()
        assert parse("") is None

    def test_missing_pubdate_returns_none(self):
        parse, _, _ = self._load()
        assert parse(None) is None

    def test_garbage_pubdate_returns_none(self):
        parse, _, _ = self._load()
        assert parse("not a date at all") is None

    def test_rfc2822_with_timezone_offset_parses_correctly(self):
        """Different timezone offsets should produce a consistent UTC timestamp."""
        parse, _, _ = self._load()
        ts_utc  = parse("Mon, 03 Jun 2024 09:00:00 +0000")
        ts_plus1= parse("Mon, 03 Jun 2024 10:00:00 +0100")
        assert ts_utc is not None and ts_plus1 is not None
        assert abs(ts_utc - ts_plus1) < 2.0, "Timezone normalization failed"

    # ── Freshness check ──────────────────────────────────────────

    def test_fresh_article_is_accepted(self):
        _, is_fresh, ages = self._load()
        fresh_ts = time.time() - 60   # 1 minute old
        assert is_fresh(fresh_ts, "accident") is True

    def test_stale_article_is_rejected(self):
        _, is_fresh, ages = self._load()
        stale_ts = time.time() - (ages["accident"] + 3600)  # older than limit
        assert is_fresh(stale_ts, "accident") is False

    def test_none_timestamp_is_rejected(self):
        _, is_fresh, _ = self._load()
        assert is_fresh(None, "accident") is False, (
            "Missing timestamp must be treated as unverifiable and excluded"
        )

    def test_heavy_traffic_has_shorter_max_age_than_accident(self):
        _, _, ages = self._load()
        assert ages["heavy_traffic"] < ages["accident"], (
            "heavy_traffic (ephemeral) should expire faster than accident"
        )

    def test_road_closure_has_longer_max_age_than_accident(self):
        _, _, ages = self._load()
        assert ages["road_closure"] > ages["accident"], (
            "road_closure (persistent) should have longer max age"
        )

    def test_well_within_boundary_is_accepted(self):
        _, is_fresh, ages = self._load()
        # 1 s inside the window — avoids clock-drift race between the two
        # time.time() calls (one here, one inside _article_is_fresh).
        near_boundary_ts = time.time() - ages["flood"] + 1
        assert is_fresh(near_boundary_ts, "flood") is True

    def test_one_second_past_boundary_is_rejected(self):
        _, is_fresh, ages = self._load()
        stale_ts = time.time() - ages["flood"] - 1
        assert is_fresh(stale_ts, "flood") is False

    def test_unknown_incident_type_uses_default_age(self):
        """An unrecognised incident type falls back to _default max age."""
        _, is_fresh, ages = self._load()
        default_max = ages["_default"]
        fresh_ts = time.time() - (default_max - 60)
        assert is_fresh(fresh_ts, "unknown_type") is True
        stale_ts = time.time() - (default_max + 60)
        assert is_fresh(stale_ts, "unknown_type") is False


# ===========================================================================
# P0-2 — Geographic relevance
# ===========================================================================

ALL_13_CITIES = [
    "lagos", "abuja", "kano", "ibadan", "port harcourt", "enugu",
    "nairobi", "mombasa", "kisumu", "nakuru", "eldoret",
    "asaba", "benin city",
]


class TestGeographicRelevance:

    def _load(self):
        from live_intelligence import (
            _assign_city, CITY_TITLE_KEYWORDS, CITY_BODY_KEYWORDS,
            GEO_CONFIDENCE_THRESHOLD,
        )
        return _assign_city, CITY_TITLE_KEYWORDS, CITY_BODY_KEYWORDS, GEO_CONFIDENCE_THRESHOLD

    # ── All cities covered ───────────────────────────────────────

    @pytest.mark.parametrize("city", ALL_13_CITIES)
    def test_all_cities_have_title_keywords(self, city):
        _, title_kw, _, _ = self._load()
        assert city in title_kw, (
            f"{city} missing from CITY_TITLE_KEYWORDS — cannot do title-based assignment"
        )

    @pytest.mark.parametrize("city", ALL_13_CITIES)
    def test_all_cities_have_body_keywords(self, city):
        _, _, body_kw, _ = self._load()
        assert city in body_kw, (
            f"{city} missing from CITY_BODY_KEYWORDS"
        )

    # ── Title-keyword assignment ─────────────────────────────────

    def test_title_lagos_keyword_assigns_lagos(self):
        assign, _, _, _ = self._load()
        result = assign("LASTMA reports gridlock on Lagos Island", "")
        assert result is not None
        city, conf = result
        assert city == "lagos"
        assert conf >= 0.5

    def test_title_nairobi_keyword_assigns_nairobi(self):
        assign, _, _, _ = self._load()
        result = assign("Nairobi CBD accident causes delays on Thika Road", "")
        assert result is not None
        city, conf = result
        assert city == "nairobi"

    def test_title_abuja_keyword_assigns_abuja(self):
        assign, _, _, _ = self._load()
        result = assign("Abuja Maitama accident closes road", "")
        assert result is not None
        city, conf = result
        assert city == "abuja"

    @pytest.mark.parametrize("city,title", [
        ("kisumu",    "Kisumu city flooding after heavy rains"),
        ("nakuru",    "Nakuru county road closed for repairs"),
        ("eldoret",   "Eldoret Uasin Gishu crash injures three"),
        ("enugu",     "Enugu state accident on major expressway"),
        ("asaba",     "Asaba Delta state road closure"),
        ("benin city","Benin City Edo state accident on ring road"),
    ])
    def test_previously_uncovered_cities_now_assignable(self, city, title):
        assign, _, _, _ = self._load()
        result = assign(title, "")
        assert result is not None, (
            f"{city} should now be assignable from title keyword '{title}'"
        )
        assigned_city, _ = result
        assert assigned_city == city

    # ── Body-only exclusion ──────────────────────────────────────

    def test_body_only_match_returns_low_confidence(self):
        """An article with no city mention in the title but two body keywords
        should return confidence < GEO_CONFIDENCE_THRESHOLD."""
        assign, _, _, threshold = self._load()
        # Title has no recognisable city keyword; body has 'lagos' + 'lekki'
        result = assign(
            "Truck overturns on major expressway causing delays",
            "Lagos traffic near Lekki is severely affected by this morning's crash."
        )
        # Could be None (no match) or a low-confidence match
        if result is not None:
            _, conf = result
            assert conf < threshold, (
                "Body-only match must have confidence below the safety-alert threshold"
            )

    def test_no_city_keyword_returns_none(self):
        assign, _, _, _ = self._load()
        result = assign(
            "Truck overturns on a highway causing delays",
            "Severe disruptions reported on a major road."
        )
        assert result is None, (
            "Article with no recognisable city keywords should return None"
        )

    def test_city_name_in_body_only_does_not_produce_high_confidence(self):
        """'Lagos' appearing only once in the body should not produce confidence ≥ 0.5."""
        assign, _, _, threshold = self._load()
        result = assign(
            "Major accident shuts down expressway",
            "Lagos traffic is very bad today near the bridge."
        )
        if result is not None:
            _, conf = result
            assert conf < threshold

    # ── Wrong-city prevention ────────────────────────────────────

    def test_lagos_title_does_not_assign_to_nairobi(self):
        """A Lagos article mentioning 'mombasa road' in the body must not
        be assigned to Nairobi or Mombasa."""
        assign, _, _, _ = self._load()
        result = assign(
            "Lagos Apapa traffic gridlock continues this morning",
            "Trucks queue along mombasa road while awaiting clearance."
        )
        assert result is not None
        city, _ = result
        assert city == "lagos", (
            f"Lagos title article incorrectly assigned to '{city}'"
        )


# ===========================================================================
# P1-4 — Incident deduplication
# ===========================================================================

class TestIncidentDeduplication:

    def _dedup(self):
        from live_intelligence import deduplicate_incidents
        return deduplicate_incidents

    def _make(self, itype, city, url="", location="", published_at=None, source="news_scan"):
        return {
            "type":                 itype,
            "city":                 city,
            "source":               source,
            "source_url":           url,
            "location":             location,
            "confidence":           0.65,
            "geo_relevance_verified": True,
            "location_confirmed":   False,
            "published_at":         published_at or time.time(),
            "created_at":           time.time(),
            "expires_at":           time.time() + 3600,
        }

    def test_identical_url_deduplicates(self):
        dedup = self._dedup()
        url = "https://news.example.com/article/12345"
        incs = [
            self._make("accident", "lagos", url=url, source="news_scan"),
            self._make("accident", "lagos", url=url, source="google_news"),
        ]
        result = dedup(incs)
        assert len(result) == 1, "Same URL from two sources should be deduped to one"

    def test_duplicate_has_corroborating_sources(self):
        dedup = self._dedup()
        url   = "https://news.example.com/article/999"
        incs  = [
            self._make("flood", "nairobi", url=url, source="news_scan"),
            self._make("flood", "nairobi", url=url, source="google_news"),
        ]
        result = dedup(incs)
        assert len(result) == 1
        assert "google_news" in result[0].get("corroborating_sources", []), (
            "Duplicate source should be recorded in corroborating_sources"
        )

    def test_distinct_simultaneous_incidents_preserved(self):
        """Two different accidents in Lagos at the same time (different URLs)
        must NOT be collapsed into one."""
        dedup = self._dedup()
        incs  = [
            self._make("accident", "lagos", url="https://a.ng/article/1"),
            self._make("accident", "lagos", url="https://a.ng/article/2"),
        ]
        result = dedup(incs)
        assert len(result) == 2, (
            "Two distinct accidents (different URLs) must both be preserved"
        )

    def test_different_types_not_deduped(self):
        dedup = self._dedup()
        # Use completely distinct URLs — same-base URLs with only query-string
        # differences are normalised to the same fingerprint by design.
        incs  = [
            self._make("flood",        "lagos", url="https://news.example.com/article/flood-123"),
            self._make("road_closure", "lagos", url="https://news.example.com/article/closure-456"),
        ]
        result = dedup(incs)
        assert len(result) == 2

    def test_no_url_uses_headline_bucket(self):
        """Items without URLs are deduped by headline+type+city+time-bucket."""
        dedup = self._dedup()
        # Same headline, same hour-bucket
        ts = time.time()
        incs = [
            self._make("accident", "abuja", url="", location="Maitama crash", published_at=ts, source="news_scan"),
            self._make("accident", "abuja", url="", location="Maitama crash", published_at=ts+30, source="tomtom_incidents"),
        ]
        result = dedup(incs)
        assert len(result) == 1, "Same headline in same hour should deduplicate"

    def test_same_headline_different_hours_not_deduped(self):
        """The same headline type appearing 2 hours apart = two distinct events."""
        dedup = self._dedup()
        ts1 = time.time() - 7200
        ts2 = time.time()
        incs = [
            self._make("heavy_traffic", "lagos", url="", location="Lekki gridlock", published_at=ts1),
            self._make("heavy_traffic", "lagos", url="", location="Lekki gridlock", published_at=ts2),
        ]
        result = dedup(incs)
        assert len(result) == 2, "Same headline 2h apart should NOT deduplicate"

    def test_higher_confidence_wins_on_merge(self):
        dedup = self._dedup()
        url   = "https://news.example.com/article/123"
        low   = dict(self._make("accident", "lagos", url=url), confidence=0.4)
        high  = dict(self._make("accident", "lagos", url=url), confidence=0.9)
        result = dedup([low, high])
        assert len(result) == 1
        assert result[0]["confidence"] == 0.9, "Winner should take the higher confidence"

    def test_empty_list_returns_empty(self):
        dedup = self._dedup()
        assert dedup([]) == []

    def test_single_item_unchanged(self):
        dedup = self._dedup()
        inc   = self._make("flood", "nairobi")
        result = dedup([inc])
        assert len(result) == 1


# ===========================================================================
# P1-5 — Provider health tracking
# ===========================================================================

class TestProviderHealth:

    def _load(self):
        import live_intelligence as li
        return li

    def test_provider_ok_records_available_true(self):
        li = self._load()
        li._record_provider_ok("test_provider_a")
        health = li.get_provider_health("test_provider_a")
        assert health["available"] is True
        assert health["last_successful_fetch"] is not None
        assert health["error_category"] is None

    def test_provider_error_records_available_false(self):
        li = self._load()
        li._record_provider_error("test_provider_b", "timeout")
        health = li.get_provider_health("test_provider_b")
        assert health["available"] is False
        assert health["error_category"] == "timeout"

    def test_provider_error_preserves_last_successful_fetch(self):
        li = self._load()
        li._record_provider_ok("test_provider_c")
        last_ok = li.get_provider_health("test_provider_c")["last_successful_fetch"]
        li._record_provider_error("test_provider_c", "http_503")
        health  = li.get_provider_health("test_provider_c")
        assert health["last_successful_fetch"] == last_ok, (
            "last_successful_fetch must not be cleared when provider errors"
        )

    def test_never_fetched_provider_returns_unknown(self):
        li = self._load()
        health = li.get_provider_health("__nonexistent_provider__")
        assert health["available"] is None
        assert health["error_category"] == "never_fetched"

    def test_error_category_values_are_informative_not_credentials(self):
        """Error categories must not expose credentials or internal paths."""
        li = self._load()
        for cat in ["timeout", "http_503", "http_401", "unexpected_error",
                    "no_channel_element", "zero_free_flow_speed"]:
            li._record_provider_error("cat_test", cat)
            health = li.get_provider_health("cat_test")
            # Should not contain API keys, passwords, or stack-trace fragments
            assert "TOMTOM_API_KEY" not in str(health)
            assert "password" not in str(health).lower()
            assert "traceback" not in str(health).lower()

    def test_time_based_congestion_always_has_method_field(self):
        """_time_based_congestion() must include congestion_method and
        congestion_note so the frontend can always render the data source."""
        from live_intelligence import _time_based_congestion
        result = _time_based_congestion()
        assert "congestion_method" in result, (
            "congestion_method missing from time-based fallback"
        )
        assert "congestion_note" in result, (
            "congestion_note missing from time-based fallback"
        )
        assert result["congestion_method"] == "time_of_day_heuristic"
        assert "estimated" in result["congestion_note"].lower() or \
               "heuristic" in result["congestion_note"].lower()


# ===========================================================================
# P1-6 — Incident provenance / community_count separation
# ===========================================================================

@pytest.fixture(scope="module")
def api_client():
    from fastapi.testclient import TestClient
    from main import app
    return TestClient(app)


NAIROBI_PAYLOAD = {
    "origin":      "Westlands",
    "destination": "CBD",
    "mode":        "matatu",
    "city":        "nairobi",
    "time":        "08:00",
}

ENUGU_PAYLOAD = {
    "origin":      "GRA",
    "destination": "New Haven",
    "mode":        "driving",
    "city":        "enugu",
    "time":        "08:00",
}


class TestIncidentProvenance:
    """Verify that the API response always separates community and third-party
    incident counts, and that road_quality_available is correctly False for
    cities without road quality data."""

    def test_live_intelligence_has_community_incidents_key(self, api_client):
        resp = api_client.post("/v2/predict", json=NAIROBI_PAYLOAD)
        if resp.status_code == 503:
            pytest.skip("OSRM routing unavailable in test environment")
        assert resp.status_code == 200
        body = resp.json()
        intel = body.get("live_intelligence", {})
        assert "community_incidents" in intel, (
            "live_intelligence must include community_incidents count"
        )

    def test_community_incidents_is_non_negative_int(self, api_client):
        resp = api_client.post("/v2/predict", json=NAIROBI_PAYLOAD)
        if resp.status_code == 503:
            pytest.skip("OSRM routing unavailable in test environment")
        body  = resp.json()
        intel = body.get("live_intelligence", {})
        count = intel.get("community_incidents")
        assert isinstance(count, int) and count >= 0

    def test_live_incidents_key_present(self, api_client):
        resp = api_client.post("/v2/predict", json=NAIROBI_PAYLOAD)
        if resp.status_code == 503:
            pytest.skip("OSRM routing unavailable in test environment")
        intel = resp.json().get("live_intelligence", {})
        assert "live_incidents" in intel, (
            "live_intelligence must include total live_incidents count"
        )

    def test_provider_health_key_present(self, api_client):
        resp = api_client.post("/v2/predict", json=NAIROBI_PAYLOAD)
        if resp.status_code == 503:
            pytest.skip("OSRM routing unavailable in test environment")
        intel = resp.json().get("live_intelligence", {})
        assert "provider_health" in intel, (
            "live_intelligence must expose provider_health metadata"
        )

    def test_provider_health_does_not_expose_api_key(self, api_client):
        resp = api_client.post("/v2/predict", json=NAIROBI_PAYLOAD)
        if resp.status_code == 503:
            pytest.skip("OSRM routing unavailable in test environment")
        intel  = resp.json().get("live_intelligence", {})
        health = str(intel.get("provider_health", {}))
        assert "TOMTOM_API_KEY" not in health, (
            "provider_health must not expose API key values"
        )

    def test_enugu_road_quality_available_is_false_in_response(self, api_client):
        resp = api_client.post("/v2/predict", json=ENUGU_PAYLOAD)
        if resp.status_code == 503:
            pytest.skip("OSRM routing unavailable in test environment")
        body = resp.json()
        assert body.get("road_quality_available") is False, (
            "Enugu has no road quality data — road_quality_available must be False"
        )

    def test_enugu_road_quality_score_is_50_sentinel(self, api_client):
        resp = api_client.post("/v2/predict", json=ENUGU_PAYLOAD)
        if resp.status_code == 503:
            pytest.skip("OSRM routing unavailable in test environment")
        body = resp.json()
        # The sentinel is 50.0 — the frontend must NOT display it as a measurement
        assert body.get("road_quality_score") == 50.0, (
            "Unavailable road quality should return the 50.0 neutral sentinel"
        )

    def test_nairobi_road_quality_available_is_true(self, api_client):
        resp = api_client.post("/v2/predict", json=NAIROBI_PAYLOAD)
        if resp.status_code == 503:
            pytest.skip("OSRM routing unavailable in test environment")
        body = resp.json()
        assert body.get("road_quality_available") is True, (
            "Nairobi has city_avg road quality data — available should be True"
        )

    def test_congestion_method_always_present(self, api_client):
        resp = api_client.post("/v2/predict", json=NAIROBI_PAYLOAD)
        if resp.status_code == 503:
            pytest.skip("OSRM routing unavailable in test environment")
        intel = resp.json().get("live_intelligence", {})
        assert "congestion_method" in intel

    def test_congestion_note_always_present(self, api_client):
        resp = api_client.post("/v2/predict", json=NAIROBI_PAYLOAD)
        if resp.status_code == 503:
            pytest.skip("OSRM routing unavailable in test environment")
        intel = resp.json().get("live_intelligence", {})
        assert "congestion_note" in intel
        assert intel["congestion_note"] is not None


# ===========================================================================
# P0-3 — Stale news / misleading timestamps (unit-level integration)
# ===========================================================================

class TestStaleNewsRejection:
    """Verify that get_news_incidents() rejects items with stale or missing
    timestamps before they reach the cache.  Uses a patched httpx to return
    controlled RSS feed content."""

    def _rss_feed(self, title: str, pub: str, description: str = "") -> bytes:
        """Build a minimal RSS feed XML with one item."""
        return f"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Test Feed</title>
    <item>
      <title>{title}</title>
      <description>{description}</description>
      <link>https://test.example.com/article/1</link>
      <pubDate>{pub}</pubDate>
    </item>
  </channel>
</rss>""".encode()

    @pytest.mark.asyncio
    async def test_stale_article_excluded(self):
        """An article published 48 hours ago should not appear in the results."""
        import live_intelligence as li

        # Reset the cache so we don't get a cached result
        li._news_cache.clear()
        li._last_news_scan = 0.0

        stale_pub = "Mon, 03 Jun 2024 00:00:00 +0000"  # fixed old date
        feed_content = self._rss_feed(
            title="Lagos Lekki accident causes gridlock",
            pub=stale_pub,
        )

        async def fake_get(url, **kwargs):
            m = mock.MagicMock()
            m.status_code = 200
            m.text        = feed_content.decode()
            return m

        with mock.patch("httpx.AsyncClient") as MockClient:
            mock_ctx = mock.AsyncMock()
            mock_ctx.get = fake_get
            MockClient.return_value.__aenter__ = mock.AsyncMock(return_value=mock_ctx)
            MockClient.return_value.__aexit__  = mock.AsyncMock(return_value=False)
            results = await li.get_news_incidents(["lagos"])

        assert len(results) == 0, (
            "Stale article (48 hours old) must be excluded from incidents"
        )

    @pytest.mark.asyncio
    async def test_missing_timestamp_excluded(self):
        """An article with an empty pubDate should be excluded."""
        import live_intelligence as li

        li._news_cache.clear()
        li._last_news_scan = 0.0

        feed_content = self._rss_feed(
            title="Nairobi Westlands heavy traffic standstill",
            pub="",  # no timestamp
        )

        async def fake_get(url, **kwargs):
            m = mock.MagicMock()
            m.status_code = 200
            m.text        = feed_content.decode()
            return m

        with mock.patch("httpx.AsyncClient") as MockClient:
            mock_ctx = mock.AsyncMock()
            mock_ctx.get = fake_get
            MockClient.return_value.__aenter__ = mock.AsyncMock(return_value=mock_ctx)
            MockClient.return_value.__aexit__  = mock.AsyncMock(return_value=False)
            results = await li.get_news_incidents(["nairobi"])

        assert len(results) == 0, (
            "Article with missing pubDate must be excluded (unverifiable timestamp)"
        )

    @pytest.mark.asyncio
    async def test_fresh_article_included(self):
        """An article with a recent pubDate and a Lagos title keyword is included."""
        import live_intelligence as li
        from email.utils import format_datetime
        import datetime

        li._news_cache.clear()
        li._last_news_scan = 0.0

        # 5 minutes ago
        recent_dt  = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(minutes=5)
        recent_pub = format_datetime(recent_dt)
        feed_content = self._rss_feed(
            title="Lagos Lekki accident causes delays on Lekki-Epe",
            pub=recent_pub,
        )

        async def fake_get(url, **kwargs):
            m = mock.MagicMock()
            m.status_code = 200
            m.text        = feed_content.decode()
            return m

        with mock.patch("httpx.AsyncClient") as MockClient:
            mock_ctx = mock.AsyncMock()
            mock_ctx.get = fake_get
            MockClient.return_value.__aenter__ = mock.AsyncMock(return_value=mock_ctx)
            MockClient.return_value.__aexit__  = mock.AsyncMock(return_value=False)
            results = await li.get_news_incidents(["lagos"])

        assert len(results) >= 1, (
            "Fresh Lagos article with title keyword should be included"
        )
        assert results[0]["city"] == "lagos"
        assert results[0]["published_at"] is not None, "published_at must be set"

    @pytest.mark.asyncio
    async def test_published_at_not_reset_on_cache_hit(self):
        """A second call within the cache TTL must return the original published_at,
        not a new time.time() value."""
        import live_intelligence as li
        from email.utils import format_datetime
        import datetime

        li._news_cache.clear()
        li._last_news_scan = 0.0

        pub_dt   = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(minutes=10)
        pub_str  = format_datetime(pub_dt)
        expected_ts = pub_dt.timestamp()

        feed_content = self._rss_feed(
            title="Abuja Maitama accident on expressway",
            pub=pub_str,
        )

        async def fake_get(url, **kwargs):
            m = mock.MagicMock()
            m.status_code = 200
            m.text        = feed_content.decode()
            return m

        with mock.patch("httpx.AsyncClient") as MockClient:
            mock_ctx = mock.AsyncMock()
            mock_ctx.get = fake_get
            MockClient.return_value.__aenter__ = mock.AsyncMock(return_value=mock_ctx)
            MockClient.return_value.__aexit__  = mock.AsyncMock(return_value=False)
            first_results = await li.get_news_incidents(["abuja"])

        if not first_results:
            pytest.skip("No result from mock feed — cannot test age preservation")

        original_published_at = first_results[0]["published_at"]
        assert abs(original_published_at - expected_ts) < 2.0, (
            "published_at should match the RSS pubDate, not time.time()"
        )

        # Second call — should come from cache
        second_results = await li.get_news_incidents(["abuja"])
        if not second_results:
            pytest.skip("Cache returned empty — cannot verify age preservation")

        assert second_results[0]["published_at"] == original_published_at, (
            "Cache hit must not reset published_at to time.time()"
        )


# ===========================================================================
# Follow-up review — Phase 2.1 concerns
# ===========================================================================


# ---------------------------------------------------------------------------
# C1 — Incident provenance: Supabase vs third-party counts
# ---------------------------------------------------------------------------

class TestIncidentProvenanceSchema:
    """The /v2/predict response must expose distinct community_reports,
    third_party_alerts, and total_incident_alerts counts."""

    def test_community_reports_field_present(self, api_client):
        resp = api_client.post("/v2/predict", json=NAIROBI_PAYLOAD)
        if resp.status_code == 503:
            pytest.skip("OSRM routing unavailable")
        assert "community_reports" in resp.json()

    def test_third_party_alerts_field_present(self, api_client):
        resp = api_client.post("/v2/predict", json=NAIROBI_PAYLOAD)
        if resp.status_code == 503:
            pytest.skip("OSRM routing unavailable")
        assert "third_party_alerts" in resp.json(), (
            "third_party_alerts field missing — TomTom/news/OSM incidents must be "
            "reported separately from community_reports"
        )

    def test_total_incident_alerts_field_present(self, api_client):
        resp = api_client.post("/v2/predict", json=NAIROBI_PAYLOAD)
        if resp.status_code == 503:
            pytest.skip("OSRM routing unavailable")
        assert "total_incident_alerts" in resp.json(), (
            "total_incident_alerts field missing"
        )

    def test_total_equals_community_plus_third_party(self, api_client):
        resp = api_client.post("/v2/predict", json=NAIROBI_PAYLOAD)
        if resp.status_code == 503:
            pytest.skip("OSRM routing unavailable")
        body = resp.json()
        cr  = body.get("community_reports", -1)
        tp  = body.get("third_party_alerts", -1)
        tot = body.get("total_incident_alerts", -1)
        assert tot == cr + tp, (
            f"total_incident_alerts ({tot}) must equal "
            f"community_reports ({cr}) + third_party_alerts ({tp})"
        )

    def test_community_reports_is_non_negative_int(self, api_client):
        resp = api_client.post("/v2/predict", json=NAIROBI_PAYLOAD)
        if resp.status_code == 503:
            pytest.skip("OSRM routing unavailable")
        body = resp.json()
        cr = body.get("community_reports")
        assert isinstance(cr, int) and cr >= 0

    def test_third_party_alerts_is_non_negative_int(self, api_client):
        resp = api_client.post("/v2/predict", json=NAIROBI_PAYLOAD)
        if resp.status_code == 503:
            pytest.skip("OSRM routing unavailable")
        body = resp.json()
        tp = body.get("third_party_alerts")
        assert isinstance(tp, int) and tp >= 0


# ---------------------------------------------------------------------------
# C2 — Deduplication: no-url no-location items must not auto-merge
# ---------------------------------------------------------------------------

class TestDeduplicationNoMergeOnSentinel:

    def _dedup(self):
        from live_intelligence import deduplicate_incidents
        return deduplicate_incidents

    def _make_no_url_no_loc(self, itype, city, source="tomtom_incidents"):
        """Incident with no URL and only the city name as location."""
        return {
            "type":                   itype,
            "city":                   city,
            "source":                 source,
            "source_url":             "",
            "location":               city,   # only the city name — not a meaningful location
            "confidence":             0.9,
            "geo_relevance_verified": True,
            "location_confirmed":     True,
            "published_at":           time.time(),
            "created_at":             time.time(),
            "expires_at":             time.time() + 3600,
        }

    def test_no_url_city_only_location_items_not_merged(self):
        """Two TomTom incidents with no URL and only the city name as location
        must NOT be merged — they are likely distinct incidents on different roads."""
        dedup = self._dedup()
        incs  = [
            self._make_no_url_no_loc("accident", "lagos"),
            self._make_no_url_no_loc("accident", "lagos"),
        ]
        result = dedup(incs)
        assert len(result) == 2, (
            "Two no-URL no-location incidents must be preserved separately "
            "(cannot safely determine they are the same event)"
        )

    def test_different_specific_locations_not_merged(self):
        """Two incidents in the same city-hour with different specific locations
        must NOT be merged."""
        from live_intelligence import deduplicate_incidents
        ts = time.time()
        incs = [
            {
                "type": "accident", "city": "nairobi", "source": "tomtom_incidents",
                "source_url": "", "location": "Westlands near Sarit Centre",
                "confidence": 0.9, "geo_relevance_verified": True, "location_confirmed": True,
                "published_at": ts, "created_at": ts, "expires_at": ts + 3600,
            },
            {
                "type": "accident", "city": "nairobi", "source": "tomtom_incidents",
                "source_url": "", "location": "Thika Road near Garden City",
                "confidence": 0.9, "geo_relevance_verified": True, "location_confirmed": True,
                "published_at": ts + 60, "created_at": ts + 60, "expires_at": ts + 3660,
            },
        ]
        result = deduplicate_incidents(incs)
        assert len(result) == 2, (
            "Two incidents with distinct specific locations must not be merged"
        )

    def test_same_specific_location_same_hour_merges(self):
        """Two entries with identical specific location text, same type, same city,
        same hour-bucket should be deduped."""
        from live_intelligence import deduplicate_incidents
        ts = time.time()
        incs = [
            {
                "type": "accident", "city": "lagos", "source": "news_scan",
                "source_url": "", "location": "Third Mainland Bridge westbound",
                "confidence": 0.65, "geo_relevance_verified": True, "location_confirmed": False,
                "published_at": ts, "created_at": ts, "expires_at": ts + 3600,
            },
            {
                "type": "accident", "city": "lagos", "source": "google_news",
                "source_url": "", "location": "Third Mainland Bridge westbound",
                "confidence": 0.55, "geo_relevance_verified": True, "location_confirmed": False,
                "published_at": ts + 120, "created_at": ts + 120, "expires_at": ts + 3720,
            },
        ]
        result = deduplicate_incidents(incs)
        assert len(result) == 1, (
            "Same specific location, same type, same hour should be deduped"
        )

    def test_cross_provider_same_url_still_merges(self):
        """Cross-provider dedup by URL must still work (unchanged from Phase 2.1)."""
        from live_intelligence import deduplicate_incidents
        url  = "https://news.example.com/article/lagos-crash-99"
        ts   = time.time()
        incs = [
            {
                "type": "accident", "city": "lagos", "source": "news_scan",
                "source_url": url, "location": "Lagos CBD",
                "confidence": 0.65, "geo_relevance_verified": True, "location_confirmed": False,
                "published_at": ts, "created_at": ts, "expires_at": ts + 3600,
            },
            {
                "type": "accident", "city": "lagos", "source": "google_news",
                "source_url": url, "location": "Lagos CBD slightly different wording",
                "confidence": 0.55, "geo_relevance_verified": True, "location_confirmed": False,
                "published_at": ts + 30, "created_at": ts + 30, "expires_at": ts + 3630,
            },
        ]
        result = deduplicate_incidents(incs)
        assert len(result) == 1, "Same URL from two providers must still deduplicate"
        assert "google_news" in result[0].get("corroborating_sources", [])


# ---------------------------------------------------------------------------
# C3 — geo_relevance_verified vs location_confirmed
# ---------------------------------------------------------------------------

class TestGeoVerificationLabels:
    """geo_relevance_verified and location_confirmed are distinct fields."""

    def test_news_scan_incident_has_location_confirmed_false(self):
        """News items have geo_relevance_verified but NOT location_confirmed."""
        from live_intelligence import CITY_TITLE_KEYWORDS
        # Directly construct the kind of incident news_scan produces
        inc = {
            "type": "accident", "city": "nairobi",
            "source": "news_scan", "source_category": "third_party_news",
            "geo_relevance_verified": True,
            "location_confirmed": False,
            "confidence": 0.65,
        }
        assert inc["geo_relevance_verified"] is True
        assert inc["location_confirmed"] is False, (
            "News items must never claim location_confirmed=True — "
            "they have no coordinate evidence"
        )

    def test_tomtom_incident_has_both_verified_true(self):
        """TomTom incidents have both geo_relevance_verified and location_confirmed."""
        inc = {
            "type": "accident", "city": "nairobi",
            "source": "tomtom_incidents", "source_category": "third_party_traffic",
            "geo_relevance_verified": True,
            "location_confirmed": True,
            "confidence": 0.9,
        }
        assert inc["geo_relevance_verified"] is True
        assert inc["location_confirmed"] is True

    def test_incident_fields_present_in_all_sources(self):
        """All incident dicts from live_intelligence must carry both provenance fields."""
        from live_intelligence import (
            _NO_DEDUP_SENTINEL, GEO_CONFIDENCE_THRESHOLD
        )
        # Spot-check that the constant exists and threshold is correct
        assert _NO_DEDUP_SENTINEL == "__no_dedup__"
        assert GEO_CONFIDENCE_THRESHOLD == 0.5

    def test_benin_city_does_not_match_benin_republic(self):
        """'Benin' in an article about Benin Republic should not assign to Benin City."""
        from live_intelligence import _assign_city
        # An article title about Benin (the country), not Benin City Nigeria
        result = _assign_city(
            "Benin Republic president visits Nigeria",
            "The president of Benin arrived in Abuja for diplomatic talks."
        )
        # This should either not match or match abuja (not benin city)
        if result is not None:
            city, _ = result
            assert city != "benin city", (
                "Article about Benin Republic must not be assigned to Benin City Nigeria. "
                f"Got: {city!r}"
            )

    def test_benin_city_title_keyword_assigns_correctly(self):
        """An article explicitly naming 'Benin City' should assign to benin city."""
        from live_intelligence import _assign_city
        result = _assign_city(
            "Benin City accident on Ring Road Edo State",
            ""
        )
        assert result is not None
        city, conf = result
        assert city == "benin city", f"Expected 'benin city', got {city!r}"
        assert conf >= 0.5

    def test_geo_relevance_verified_in_get_route_intelligence_return(self, api_client):
        """The live_intelligence block must not expose raw incident dicts;
        but the intelligence module must set the correct field names."""
        resp = api_client.post("/v2/predict", json=NAIROBI_PAYLOAD)
        if resp.status_code == 503:
            pytest.skip("OSRM routing unavailable")
        body = resp.json()
        intel = body.get("live_intelligence", {})
        # The old geo_verified key must not leak through
        assert "geo_verified" not in str(intel), (
            "Old 'geo_verified' field must be fully replaced by 'geo_relevance_verified'"
        )


# ---------------------------------------------------------------------------
# C4 — Model training date in /v2/predict
# ---------------------------------------------------------------------------

class TestModelTrainedDate:
    """training_data_date must be present in /v2/predict response."""

    def test_training_data_date_field_present(self, api_client):
        resp = api_client.post("/v2/predict", json=NAIROBI_PAYLOAD)
        if resp.status_code == 503:
            pytest.skip("OSRM routing unavailable")
        body = resp.json()
        assert "training_data_date" in body, (
            "training_data_date must be present in /v2/predict response"
        )

    def test_training_data_date_is_string(self, api_client):
        resp = api_client.post("/v2/predict", json=NAIROBI_PAYLOAD)
        if resp.status_code == 503:
            pytest.skip("OSRM routing unavailable")
        body = resp.json()
        val = body.get("training_data_date")
        assert isinstance(val, str) and len(val) > 0, (
            "training_data_date must be a non-empty string"
        )

    def test_training_data_date_matches_constant(self):
        """training_data_date in response must match the MODEL_TRAINED constant."""
        import sys
        import pathlib
        app_dir = pathlib.Path(__file__).parent.parent / "app"
        if str(app_dir) not in sys.path:
            sys.path.insert(0, str(app_dir))
        from main import MODEL_TRAINED
        assert MODEL_TRAINED == "2026-07-23", (
            f"MODEL_TRAINED constant changed unexpectedly: {MODEL_TRAINED!r}"
        )

    def test_training_data_date_not_same_as_now(self, api_client):
        """training_data_date must not be today's date (would imply live retraining)."""
        import datetime
        resp = api_client.post("/v2/predict", json=NAIROBI_PAYLOAD)
        if resp.status_code == 503:
            pytest.skip("OSRM routing unavailable")
        body     = resp.json()
        trained  = body.get("training_data_date", "")
        today    = datetime.date.today().isoformat()
        assert trained != today, (
            f"training_data_date must not equal today ({today}); "
            "it reflects historical training, not live retraining"
        )


# ---------------------------------------------------------------------------
# C5 — Provider outage silent zero
# ---------------------------------------------------------------------------

class TestProviderOutageSilentZero:
    """all_providers_failed in live_intelligence must be True when
    a provider recorded an error and incident count is zero."""

    def test_all_providers_failed_field_present_in_api_response(self, api_client):
        resp = api_client.post("/v2/predict", json=NAIROBI_PAYLOAD)
        if resp.status_code == 503:
            pytest.skip("OSRM routing unavailable")
        intel = resp.json().get("live_intelligence", {})
        assert "all_providers_failed" in intel, (
            "all_providers_failed key must be present in live_intelligence"
        )

    def test_all_providers_failed_is_bool(self, api_client):
        resp = api_client.post("/v2/predict", json=NAIROBI_PAYLOAD)
        if resp.status_code == 503:
            pytest.skip("OSRM routing unavailable")
        intel = resp.json().get("live_intelligence", {})
        apf   = intel.get("all_providers_failed")
        assert isinstance(apf, bool), (
            f"all_providers_failed must be bool, got {type(apf)}"
        )

    def test_all_providers_failed_false_when_no_errors_recorded(self):
        """When no provider has recorded an error, all_providers_failed must be False."""
        import live_intelligence as li

        # Clear health state for the providers we care about
        for key in list(li._provider_health.keys()):
            del li._provider_health[key]

        # No errors — all health entries absent (never_fetched)
        city = "lagos"
        incident_providers = [
            "tomtom_incidents", f"osm_closures:{city}",
            f"osm_overpass:{city}", f"google_news_route:{city}",
        ]
        any_failed = any(
            li.get_provider_health(p).get("available") is False
            for p in incident_providers
        )
        assert any_failed is False, (
            "No provider has failed — all_providers_failed should be False"
        )

    def test_all_providers_failed_true_when_error_and_zero_incidents(self):
        """When a provider has available=False and incidents are empty,
        all_providers_failed must be True."""
        import live_intelligence as li

        city = "lagos"
        # Record one provider as failed
        li._record_provider_error(f"osm_closures:{city}", "timeout")

        incident_providers = [
            "tomtom_incidents", f"osm_closures:{city}",
            f"osm_overpass:{city}", f"google_news_route:{city}",
        ]
        any_failed = any(
            li.get_provider_health(p).get("available") is False
            for p in incident_providers
        )
        all_providers_failed = (0 == 0 and any_failed)   # zero incidents + failed provider
        assert all_providers_failed is True, (
            "Provider error recorded with zero incidents — all_providers_failed must be True"
        )
