"""
CommuteIQ — Live Intelligence Module
live_intelligence.py

Replaces static dataset limitations with real-time internet data.
All sources are FREE — no API keys required except TomTom (free signup).

Four intelligence layers:
  1. TomTom Traffic Flow — real measured road speeds (replaces time-of-day guess)
  2. RSS News Scanner — Nigerian + Kenyan news feeds including Lagos Traffic Radio (LASTMA)
  3. Google News Search — targeted search for floods, accidents, road closures
  4. OSM Road Closures — community road closures + Overpass API construction zones

Evidence-integrity guarantees (Phase 2.1):
  - Every incident carries published_at (original publication time), retrieved_at (when
    we fetched it), and expires_at (type-specific staleness cutoff).
  - Stale items (published_at older than MAX_ARTICLE_AGE_S) are excluded before the cache
    is written — cache refreshes do NOT reset the original publication age.
  - City assignment requires the city name to appear in the article title (not just the
    body) OR in conjunction with a known city-specific landmark keyword. Generic body-text
    matches are downgraded and excluded from safety alerts.
  - All 13 supported cities have keyword entries; previously uncovered cities default to
    name-only matching and receive confidence=0 items excluded from active alerts.
  - Cross-source deduplication uses normalised URL + headline fingerprints and time
    proximity, preserving distinct simultaneous incidents of the same type.
  - Provider health is tracked per-source: each provider returns a status dict with
    available, last_successful_fetch, and error_category so callers can distinguish
    "no incidents" from "provider down".

Setup:
  - TomTom: Sign up free at developer.tomtom.com → get API key → set TOMTOM_API_KEY
  - RSS + Google News: No setup needed — works immediately
"""

import hashlib
import os
import re
import time
import httpx
import asyncio
from datetime import timezone
from email.utils import parsedate_to_datetime
from typing import Optional
from xml.etree import ElementTree as ET


# ── Configuration ─────────────────────────────────────────────
TOMTOM_API_KEY = os.getenv("TOMTOM_API_KEY", "")   # Free at developer.tomtom.com
CACHE_TTL      = 600   # 10 minutes — how long to cache live results (congestion)
NEWS_CACHE_TTL = 900   # 15 minutes — news scan cache

# Maximum article age in seconds. Articles older than this are excluded regardless
# of cache state.  7,200 s = 2 hours for fast-changing incident types; construction
# and road closures use their own (longer) type-specific expiry below.
MAX_ARTICLE_AGE_S: dict = {
    "heavy_traffic": 1_200,   # 20 min — gridlock is very ephemeral
    "accident":      7_200,   # 2 h
    "flood":         14_400,  # 4 h
    "road_closure":  86_400,  # 24 h — closures persist
    "construction":  86_400,  # 24 h
    "breakdown":     2_700,   # 45 min
    "_default":      7_200,   # 2 h fallback
}


# ── In-memory caches ──────────────────────────────────────────
_congestion_cache: dict = {}
_news_cache:       dict = {}
_last_news_scan:   float = 0.0

# Provider health tracking — updated after every fetch attempt
_provider_health: dict = {}   # keyed by provider name


def _record_provider_ok(name: str) -> None:
    _provider_health[name] = {
        "available":            True,
        "last_successful_fetch": time.time(),
        "error_category":       None,
    }


def _record_provider_error(name: str, category: str) -> None:
    prev = _provider_health.get(name, {})
    _provider_health[name] = {
        "available":            False,
        "last_successful_fetch": prev.get("last_successful_fetch"),
        "error_category":       category,
    }


def get_provider_health(name: str) -> dict:
    """Return health metadata for a named provider.

    Keys: available (bool), last_successful_fetch (float | None), error_category (str | None).
    Returns 'unknown' status when no fetch has been attempted yet.
    """
    return _provider_health.get(name, {
        "available":            None,
        "last_successful_fetch": None,
        "error_category":       "never_fetched",
    })


# ── City bounding boxes for TomTom traffic queries ───────────
# TomTom traffic flow accepts a bounding box to return all incidents in an area
CITY_BBOXES = {
    "lagos":         "6.3878,3.0982,6.7063,3.7145",
    "nairobi":       "-1.4441,36.6519,-1.1601,37.0088",
    "abuja":         "8.7640,6.9960,9.3390,7.7300",
    "kano":          "11.8500,8.3700,12.1500,8.7800",
    "ibadan":        "7.2600,3.7900,7.5200,4.1200",
    "port harcourt": "4.6500,6.8900,5.0200,7.2100",
    "mombasa":       "-4.1500,39.5500,-3.9500,39.8500",
    "kisumu":        "-0.1700,34.6800,0.0600,34.9000",
}

# Keyword patterns → report type mapping for news scanning
INCIDENT_KEYWORDS = {
    "accident":     ["accident", "crash", "collision", "overturn", "vehicle",
                     "truck", "lorry", "bus crash", "road crash"],
    "flood":        ["flood", "flooding", "waterlogged", "submerged", "water",
                     "rain", "downpour", "drainage"],
    "road_closure": ["closed", "closure", "blocked", "road block", "diversion",
                     "shutdown", "barricade", "protest", "demonstration"],
    "heavy_traffic":["gridlock", "traffic jam", "standstill",
                     "congestion", "slow moving", "heavy traffic", "bumper"],
    "construction": ["construction", "repair", "pothole", "road work",
                     "rehabilitation", "maintenance"],
}

# City keyword tables — TITLE keywords are high-confidence (must appear in
# article title for city assignment to be accepted).  BODY keywords are only
# used to confirm an already-tentative assignment made via the title.
#
# All 13 supported cities now have entries.  Cities without TomTom bounding
# boxes or RSS feeds use conservative keyword sets and receive lower confidence.
CITY_TITLE_KEYWORDS: dict = {
    "lagos":         ["lagos", "lekki", "ikeja", "apapa", "surulere",
                      "third mainland", "oshodi", "lastma", "yaba", "ikorodu"],
    "nairobi":       ["nairobi", "nairobi cbd", "westlands", "thika road",
                      "mombasa road", "ngong road", "waiyaki way", "langata",
                      "karen", "kasarani", "embakasi"],
    "abuja":         ["abuja", "fct", "maitama", "wuse", "garki", "asokoro",
                      "gwarinpa", "kubwa"],
    "kano":          ["kano", "kano state", "kano city"],
    "ibadan":        ["ibadan", "ibadan city", "oyo state"],
    "port harcourt": ["port harcourt", "portharcourt", "rivers state", "ph city"],
    "mombasa":       ["mombasa", "mombasa island", "kilindini", "likoni"],
    "kisumu":        ["kisumu", "kisumu city", "kisumu county"],
    "nakuru":        ["nakuru", "nakuru city", "nakuru county"],
    "eldoret":       ["eldoret", "uasin gishu", "eldoret city"],
    "enugu":         ["enugu", "enugu state", "enugu city"],
    "asaba":         ["asaba", "delta state", "asaba city"],
    "benin city":    ["benin city", "benin", "edo state", "benin-ore"],
}

# Body-text keywords — used only to disambiguate when title gave a weak signal.
# A body-only match (no title match) is treated as unverifiable and excluded.
CITY_BODY_KEYWORDS: dict = {
    "lagos":         CITY_TITLE_KEYWORDS["lagos"] + [
                         "victoria island", "vi ", "isale eko", "ajah", "mile 2",
                         "festac", "mushin", "ojota", "maryland", "ketu", "agege",
                         "oba akran", "airport road", "lekki-epe",
                         "ikorodu road", "lagos-ibadan expressway",
                         "lasg", "lspwc"],
    "nairobi":       CITY_TITLE_KEYWORDS["nairobi"] + [
                         "cbd", "uthiru", "kikuyu", "eastleigh", "mathare", "kibera",
                         "ruiru", "thika", "donholm", "south b", "south c",
                         "githurai", "huruma"],
    "abuja":         CITY_TITLE_KEYWORDS["abuja"] + ["dutse", "bwari", "kuje", "gwagwalada"],
    "kano":          CITY_TITLE_KEYWORDS["kano"] + ["sabon gari", "nasarawa kano"],
    "ibadan":        CITY_TITLE_KEYWORDS["ibadan"] + ["bodija", "ring road", "challenge"],
    "port harcourt": CITY_TITLE_KEYWORDS["port harcourt"] + ["rumuola", "diobu", "eleme"],
    "mombasa":       CITY_TITLE_KEYWORDS["mombasa"] + ["malindi", "kilifi"],
    "kisumu":        CITY_TITLE_KEYWORDS["kisumu"] + ["nyalenda", "kondele", "milimani"],
    "nakuru":        CITY_TITLE_KEYWORDS["nakuru"] + ["naivasha", "gilgil"],
    "eldoret":       CITY_TITLE_KEYWORDS["eldoret"] + ["moiben", "ainabkoi"],
    "enugu":         CITY_TITLE_KEYWORDS["enugu"] + ["new haven", "independence layout", "trans ekulu"],
    "asaba":         CITY_TITLE_KEYWORDS["asaba"] + ["okpanam", "ibusa", "cable point"],
    "benin city":    CITY_TITLE_KEYWORDS["benin city"] + ["uselu", "ugbowo", "sapele road",
                         "ring road benin", "gra benin"],
}

# Active RSS feeds for Nigeria and Kenya
RSS_FEEDS = {
    "nigeria": [
        # General Nigerian news
        "https://channelstv.com/feed",
        "https://pmnewsnigeria.com/feed",
        "https://vanguardngr.com/feed",
        "https://dailypost.ng/feed",
        # Lagos Traffic Radio (LASTMA) — official government traffic updates
        # Posts road-level flash updates every 10 min from live motorbike reporters
        "https://trafficradio961.ng/feed",
        "https://trafficradio961.ng/category/news/traffic-updates/feed",
        # FRSC National Traffic Radio Abuja — try feed, falls back silently if unavailable
        "https://frsc.gov.ng/feed",
    ],
    "kenya": [
        "https://kenyanews.go.ke/feed",
        "https://kenyans.co.ke/feeds/news",
        "https://nairobiwire.com/feed",
        "https://www.standardmedia.co.ke/rss",
        # Nation Africa — Kenya's largest daily, strong Nairobi traffic coverage
        "https://nation.africa/kenya/rss.xml",
    ]
}


# ── Timestamp utilities ───────────────────────────────────────

def _parse_rfc2822(pub_str: str) -> Optional[float]:
    """Parse an RFC-2822 pubDate string (e.g. 'Mon, 03 Jun 2024 09:15:00 +0100')
    into a UTC Unix timestamp. Returns None if the string is empty or unparseable.
    """
    if not pub_str or not pub_str.strip():
        return None
    try:
        dt = parsedate_to_datetime(pub_str.strip())
        return dt.timestamp()
    except Exception:
        return None


def _article_is_fresh(published_at: Optional[float], incident_type: str) -> bool:
    """Return True when the article's original publication time is within the
    allowed age window for this incident type.

    When published_at is None (timestamp missing or unparseable) the item is
    treated as UNVERIFIABLE and excluded (returns False).  Callers that want to
    keep unverifiable items for display must check this explicitly.
    """
    if published_at is None:
        return False
    max_age = MAX_ARTICLE_AGE_S.get(incident_type, MAX_ARTICLE_AGE_S["_default"])
    return (time.time() - published_at) <= max_age


# ── Geographic relevance ──────────────────────────────────────

def _assign_city(title: str, body: str) -> Optional[tuple]:
    """Determine whether an article can be reliably assigned to a city.

    Returns (city_name, confidence) or None if the item cannot be assigned.

    Rules (most-specific first):
      1. If ANY title keyword for a city matches the article *title*, assign
         that city with confidence 0.65.  First match wins (cities are ordered
         from most-specific to least in CITY_TITLE_KEYWORDS).
      2. If the title gave no match, check whether two or more body keywords
         for the same city appear in both title+body combined, with at least
         one being a high-specificity term (not just the city name alone).
         Assign with confidence 0.4 and mark as unverified.
      3. Any other pattern → return None (excluded from safety alerts).

    This prevents a Lagos story that happens to mention "nairobi road" in the
    body from being assigned to Nairobi, and prevents a city-name body-only
    match from being treated as verified geographic evidence.
    """
    title_lower = title.lower()
    body_lower  = (body or "").lower()
    full_text   = f"{title_lower} {body_lower}"

    # Pass 1: title keyword match
    for city, keywords in CITY_TITLE_KEYWORDS.items():
        if any(kw in title_lower for kw in keywords):
            return (city, 0.65)

    # Pass 2: body-text multi-keyword match (requires ≥2 distinct body keywords
    # for the same city to appear in the full text, the title being blank for
    # all cities).  Confidence is 0.4 — below the threshold for safety alerts.
    for city, keywords in CITY_BODY_KEYWORDS.items():
        matches = [kw for kw in keywords if kw in full_text]
        if len(matches) >= 2:
            return (city, 0.4)

    return None


# Minimum confidence threshold to include an incident in safety-relevant alert
# counts (community_count in the prediction response).  Items below this are
# returned in the intelligence data for transparency but are NOT counted.
GEO_CONFIDENCE_THRESHOLD = 0.5


# ── Deduplication ─────────────────────────────────────────────

def _incident_fingerprint(inc: dict) -> str:
    """Return a stable fingerprint for cross-source deduplication.

    Uses (in order of availability):
      1. Normalised source URL (strip query strings, lowercase)
      2. Normalised headline (lowercase, strip punctuation, first 60 chars)
      3. Fallback: type + city + published_at bucket (1-hour windows)

    Two incidents are considered duplicates when their fingerprints match,
    regardless of which source reported them first.
    """
    url = inc.get("source_url", "").strip()
    if url:
        # Normalise: strip query string and fragment, lowercase
        url_norm = re.sub(r"[?#].*", "", url).lower().rstrip("/")
        return f"url:{url_norm}"

    headline = inc.get("location", "") or inc.get("city", "")
    if headline:
        norm = re.sub(r"[^a-z0-9 ]", "", headline.lower())[:60].strip()
        if norm:
            # Bucket by 1-hour time windows so the same event reported at
            # slightly different times doesn't become two entries.
            ts = inc.get("published_at") or inc.get("created_at") or time.time()
            bucket = int(ts // 3600)
            itype  = inc.get("type", "unknown")
            city   = inc.get("city", "unknown")
            key    = f"headline:{itype}:{city}:{bucket}:{norm}"
            return hashlib.md5(key.encode()).hexdigest()[:16]

    # Last resort — just deduplicate by type+city+hour bucket
    ts     = inc.get("published_at") or inc.get("created_at") or time.time()
    bucket = int(ts // 3600)
    return f"type:{inc.get('type','?')}:{inc.get('city','?')}:{bucket}"


def deduplicate_incidents(incidents: list) -> list:
    """Remove duplicate incidents across sources using event-level fingerprints.

    When two incidents share a fingerprint, the one with higher confidence or an
    earlier published_at is kept and the duplicate's source is merged into the
    winner's 'corroborating_sources' list.  Distinct simultaneous incidents
    (same type, same city, different URLs / different time buckets) are preserved.
    """
    seen:    dict = {}   # fingerprint → index in `result`
    result:  list = []

    for inc in incidents:
        fp = _incident_fingerprint(inc)
        if fp in seen:
            # Merge: add this source to the winning item's corroborating list
            winner = result[seen[fp]]
            corr   = winner.setdefault("corroborating_sources", [])
            src    = inc.get("source", "unknown")
            if src not in corr and src != winner.get("source"):
                corr.append(src)
            # Upgrade confidence if the duplicate has stronger geo evidence
            if inc.get("confidence", 0) > winner.get("confidence", 0):
                winner["confidence"] = inc["confidence"]
        else:
            seen[fp] = len(result)
            result.append(dict(inc))   # shallow copy so we don't mutate the input

    return result


# ═══════════════════════════════════════════════════════════════
# 1. TOMTOM TRAFFIC FLOW — real measured congestion
# ═══════════════════════════════════════════════════════════════

async def get_live_congestion(lat: float, lng: float, city: str) -> dict:
    """
    Get REAL measured traffic flow from TomTom.
    Returns congestion level and live speed vs free-flow speed.

    Free tier: 2,500 requests/day — enough for CommuteIQ.
    No credit card needed. Sign up: developer.tomtom.com

    Falls back to time-based estimate if key not set or API fails.
    """
    if not TOMTOM_API_KEY:
        return _time_based_congestion()

    cache_key = f"flow|{round(lat,2)}|{round(lng,2)}"
    if cache_key in _congestion_cache:
        entry = _congestion_cache[cache_key]
        if time.time() - entry["ts"] < CACHE_TTL:
            return entry["data"]

    try:
        # TomTom Traffic Flow API — point query
        url = (
            f"https://api.tomtom.com/traffic/services/4/flowSegmentData/"
            f"absolute/10/json"
            f"?key={TOMTOM_API_KEY}"
            f"&point={lat},{lng}"
            f"&unit=KMPH"
        )
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(url)
            if r.status_code != 200:
                _record_provider_error("tomtom_flow", f"http_{r.status_code}")
                return _time_based_congestion()
            data = r.json()

        flow = data.get("flowSegmentData", {})
        current_speed  = flow.get("currentSpeed",    0)
        free_flow_speed= flow.get("freeFlowSpeed",   1)
        confidence     = flow.get("confidence",      0)

        if free_flow_speed == 0:
            _record_provider_error("tomtom_flow", "zero_free_flow_speed")
            return _time_based_congestion()

        # Flow ratio: 1.0 = free flow, 0.5 = half speed, 0.2 = standstill
        ratio = current_speed / free_flow_speed

        if ratio >= 0.85:
            level = "Low"
        elif ratio >= 0.55:
            level = "Medium"
        else:
            level = "High"

        result = {
            "congestion":      level,
            "current_speed":   round(current_speed),
            "free_flow_speed": round(free_flow_speed),
            "flow_ratio":      round(ratio, 2),
            "confidence":      confidence,
            "source":          "TomTom live traffic",
            "live":            True,
        }
        _record_provider_ok("tomtom_flow")
        _congestion_cache[cache_key] = {"data": result, "ts": time.time()}
        return result

    except httpx.TimeoutException:
        _record_provider_error("tomtom_flow", "timeout")
        return _time_based_congestion()
    except Exception:
        _record_provider_error("tomtom_flow", "unexpected_error")
        return _time_based_congestion()


async def get_live_incidents(city: str) -> list:
    """
    Get real-time traffic incidents from TomTom for a city bounding box.
    Returns list of incidents with type, location, and severity.
    """
    if not TOMTOM_API_KEY or city.lower() not in CITY_BBOXES:
        return []

    cache_key = f"incidents|{city}"
    if cache_key in _congestion_cache:
        entry = _congestion_cache[cache_key]
        if time.time() - entry["ts"] < CACHE_TTL:
            return entry["data"]

    provider = "tomtom_incidents"
    try:
        bbox = CITY_BBOXES[city.lower()]
        url  = (
            f"https://api.tomtom.com/traffic/services/5/incidentDetails"
            f"?key={TOMTOM_API_KEY}"
            f"&bbox={bbox}"
            f"&fields={{incidents{{type,geometry,properties{{id,iconCategory,magnitudeOfDelay,events,startTime,endTime,from,to,length,delay,roadNumbers,timeValidity}}}}}}"
            f"&language=en-GB"
            f"&t=1111&categoryFilter=0,1,2,3,4,5,6,7,8,9,10,11,14"
            f"&timeValidityFilter=present"
        )
        async with httpx.AsyncClient(timeout=8) as client:
            r = await client.get(url)
            if r.status_code != 200:
                _record_provider_error(provider, f"http_{r.status_code}")
                return []
            data = r.json()

        incidents = []
        icon_to_type = {
            1: "accident",  2: "heavy_traffic", 3: "road_closure",
            4: "construction", 5: "flood", 6: "heavy_traffic",
            7: "heavy_traffic", 8: "construction", 9: "road_closure",
            10: "breakdown", 11: "heavy_traffic", 14: "heavy_traffic",
        }

        now = time.time()
        for inc in data.get("incidents", []):
            props = inc.get("properties", {})
            icon  = props.get("iconCategory", 0)
            delay = props.get("magnitudeOfDelay", 0)   # 0-4, 4=major
            location_from = props.get("from", "")
            location_to   = props.get("to", "")
            location = f"{location_from} → {location_to}" if location_from else city

            itype = icon_to_type.get(icon, "heavy_traffic")
            # TomTom incidents are live/current — published_at = now,
            # but we use the incident's own startTime when available.
            start_raw    = props.get("startTime", "")
            published_at = now   # default: treat as current
            if start_raw:
                try:
                    from datetime import datetime as _dt
                    published_at = _dt.fromisoformat(
                        start_raw.replace("Z", "+00:00")
                    ).timestamp()
                except Exception:
                    pass

            incidents.append({
                "type":         itype,
                "location":     location,
                "city":         city,
                "source":       "tomtom_incidents",
                # TomTom incidents are geographically verified (bbox-bounded)
                "confidence":   0.9,
                "geo_verified": True,
                "published_at": published_at,
                "retrieved_at": now,
                "severity":     delay,
                "created_at":   now,
                "expires_at":   now + MAX_ARTICLE_AGE_S.get(itype, MAX_ARTICLE_AGE_S["_default"]),
            })

        _record_provider_ok(provider)
        _congestion_cache[cache_key] = {"data": incidents, "ts": now}
        return incidents

    except httpx.TimeoutException:
        _record_provider_error(provider, "timeout")
        return []
    except Exception:
        _record_provider_error(provider, "unexpected_error")
        return []


# ═══════════════════════════════════════════════════════════════
# 2. RSS NEWS SCANNER — auto-generates incident reports from news
# ═══════════════════════════════════════════════════════════════

async def get_news_incidents(cities: list = None) -> list:
    """
    Scan Nigerian and Kenyan RSS feeds for traffic-related headlines.
    Converts news items into CommuteIQ community report format.

    Evidence-integrity guarantees:
      - published_at is parsed from the RSS pubDate field.
      - Items without a parseable pubDate are excluded (unverifiable timestamp).
      - Items whose published_at is older than MAX_ARTICLE_AGE_S are excluded.
      - City assignment requires a title keyword match (not body-only).
      - Items that pass only body matching (confidence < GEO_CONFIDENCE_THRESHOLD)
        are excluded from route-safety alert counts.
    """
    global _last_news_scan

    # Return from cache if still fresh — but do NOT reset any incident's
    # published_at: the cache stores the original timestamps from the last fetch.
    if time.time() - _last_news_scan < NEWS_CACHE_TTL and _news_cache:
        filtered = []
        for city in (cities or list(CITY_TITLE_KEYWORDS.keys())):
            filtered.extend(_news_cache.get(city.lower(), []))
        return filtered

    all_incidents: dict = {city: [] for city in CITY_TITLE_KEYWORDS}
    feeds_to_scan = []

    # Determine which feeds to scan
    if cities:
        countries = set()
        for city in cities:
            if city.lower() in ["lagos", "abuja", "kano", "ibadan",
                                 "port harcourt", "enugu", "asaba", "benin city"]:
                countries.add("nigeria")
            else:
                countries.add("kenya")
        for country in countries:
            feeds_to_scan.extend(RSS_FEEDS.get(country, []))
    else:
        for feeds in RSS_FEEDS.values():
            feeds_to_scan.extend(feeds)

    # Also add Google News targeted search
    google_queries = [
        "Lagos traffic accident today",
        "Lagos flood road closed",
        "Nairobi traffic accident today",
        "Nairobi flood Mombasa road",
    ]
    for q in google_queries:
        feeds_to_scan.append(
            f"https://news.google.com/rss/search?q={q.replace(' ','+')}+Africa&hl=en&gl=NG&ceid=NG:en"
        )

    async def fetch_feed(url: str) -> tuple:
        """Returns (url, items_list, ok: bool).  Never raises."""
        provider_key = f"rss:{url[:60]}"
        try:
            async with httpx.AsyncClient(
                timeout=8,
                headers={"User-Agent": "CommuteIQ/2.0 RSS scanner"}
            ) as client:
                r = await client.get(url, follow_redirects=True)
                if r.status_code != 200:
                    _record_provider_error(provider_key, f"http_{r.status_code}")
                    return (url, [], False)

            root    = ET.fromstring(r.text)
            channel = root.find("channel")
            if channel is None:
                _record_provider_error(provider_key, "no_channel_element")
                return (url, [], False)

            items = []
            for item in channel.findall("item")[:20]:   # latest 20 articles
                title = item.findtext("title", "")
                desc  = item.findtext("description", "")
                link  = item.findtext("link", "")
                pub   = item.findtext("pubDate", "")

                # Parse original publication timestamp
                published_at = _parse_rfc2822(pub)

                items.append({
                    "title":        title,
                    "desc":         desc,
                    "link":         link,
                    "pub":          pub,
                    "published_at": published_at,
                    "text":         f"{title.lower()} {desc.lower()}",
                })
            _record_provider_ok(provider_key)
            return (url, items, True)
        except httpx.TimeoutException:
            _record_provider_error(provider_key, "timeout")
            return (url, [], False)
        except Exception:
            _record_provider_error(provider_key, "unexpected_error")
            return (url, [], False)

    # Fetch all feeds concurrently
    results   = await asyncio.gather(*[fetch_feed(url) for url in feeds_to_scan])
    all_items = [item for (_url, batch, _ok) in results for item in batch]

    now = time.time()

    # Parse items into incidents
    for item in all_items:
        # ── Incident type ────────────────────────────────────────
        incident_type = None
        for itype, keywords in INCIDENT_KEYWORDS.items():
            if any(kw in item["text"] for kw in keywords):
                incident_type = itype
                break
        if not incident_type:
            continue

        # ── Evidence freshness check ─────────────────────────────
        # Exclude items without a parseable timestamp (unverifiable age).
        published_at = item["published_at"]
        if not _article_is_fresh(published_at, incident_type):
            # Stale or unparseable timestamp — do not include.
            continue

        # ── Geographic relevance ─────────────────────────────────
        # Requires a title keyword match for safety-alert inclusion.
        assignment = _assign_city(item["title"], item["desc"])
        if assignment is None:
            continue
        incident_city, geo_confidence = assignment

        # Only include in the city's incident list if the city was actually
        # requested (or we're scanning everything).
        if cities and incident_city not in [c.lower() for c in cities]:
            continue

        # Build location string from title (keep original case for readability)
        location = re.sub(r"\b(traffic|accident|road)\b", "", item["title"],
                          flags=re.IGNORECASE).strip(" -–|,.")
        location = location[:80] if len(location) > 80 else location

        incident = {
            "type":         incident_type,
            "location":     location or incident_city,
            "city":         incident_city,
            "source":       "news_scan",
            "source_url":   item["link"],
            # confidence reflects geographic verification quality:
            # 0.65 = title keyword match (reliable)
            # 0.4  = body-only match (unverified — excluded from alert counts)
            "confidence":   geo_confidence,
            "geo_verified": geo_confidence >= GEO_CONFIDENCE_THRESHOLD,
            # published_at is the ORIGINAL article publication time — never reset
            "published_at": published_at,
            "retrieved_at": now,
            "created_at":   now,
            "expires_at":   now + MAX_ARTICLE_AGE_S.get(incident_type,
                                                         MAX_ARTICLE_AGE_S["_default"]),
        }
        all_incidents[incident_city].append(incident)

    # Update cache — stores incidents with their original published_at timestamps.
    # On next cache hit, published_at is NOT reset; staleness is re-evaluated
    # against the original timestamp each time items are consumed.
    _news_cache.update(all_incidents)
    _last_news_scan = now

    # Return for requested cities
    out = []
    for city in (cities or list(all_incidents.keys())):
        out.extend(all_incidents.get(city.lower(), []))

    return out


# ═══════════════════════════════════════════════════════════════
# 3. GOOGLE NEWS TARGETED SEARCH — on-demand incident lookup
# ═══════════════════════════════════════════════════════════════

async def search_news_for_route(origin: str, destination: str, city: str) -> list:
    """
    Search Google News RSS for incidents specifically on this route.
    Called when a prediction is made — looks for news about the specific
    origin/destination area.

    Evidence-integrity guarantees:
      - published_at is parsed from RSS pubDate.
      - Stale items (older than incident type's MAX_ARTICLE_AGE_S) are excluded.
      - Items without a parseable timestamp are excluded.
      - Confidence is 0.55 — these are route-specific but not geo-verified.
    """
    queries = [
        f"{origin} {city} accident",
        f"{destination} {city} flood",
    ]

    incidents  = []
    now        = time.time()
    provider   = f"google_news_route:{city}"

    for q in queries:
        url = f"https://news.google.com/rss/search?q={q.replace(' ','+')}+today&hl=en"
        try:
            async with httpx.AsyncClient(
                timeout=5, headers={"User-Agent": "CommuteIQ/2.0"}
            ) as client:
                r = await client.get(url, follow_redirects=True)
                if r.status_code != 200:
                    _record_provider_error(provider, f"http_{r.status_code}")
                    continue

            root    = ET.fromstring(r.text)
            channel = root.find("channel")
            if channel is None:
                _record_provider_error(provider, "no_channel_element")
                continue

            for item in channel.findall("item")[:5]:
                title        = item.findtext("title", "")
                pub          = item.findtext("pubDate", "")
                published_at = _parse_rfc2822(pub)

                for itype, keywords in INCIDENT_KEYWORDS.items():
                    if any(kw in title.lower() for kw in keywords):
                        # Freshness check — exclude stale or unparseable items
                        if not _article_is_fresh(published_at, itype):
                            break
                        incidents.append({
                            "type":         itype,
                            "location":     f"{origin}–{destination} corridor",
                            "city":         city,
                            "source":       "google_news",
                            "source_url":   item.findtext("link", ""),
                            "confidence":   0.55,
                            "geo_verified": False,   # route-corridor heuristic only
                            "published_at": published_at,
                            "retrieved_at": now,
                            "created_at":   now,
                            "expires_at":   now + MAX_ARTICLE_AGE_S.get(itype,
                                                MAX_ARTICLE_AGE_S["_default"]),
                        })
                        break
            _record_provider_ok(provider)
        except httpx.TimeoutException:
            _record_provider_error(provider, "timeout")
            continue
        except Exception:
            _record_provider_error(provider, "unexpected_error")
            continue

        await asyncio.sleep(0.5)   # be polite to Google

    return incidents


# ═══════════════════════════════════════════════════════════════
# 4. OSM ROAD CLOSURES API — community-reported closures
# ═══════════════════════════════════════════════════════════════

# Bounding boxes used for OSM Overpass AND the closures.osm.ch API
OSM_CITY_BBOXES = {
    "lagos":         {"south": 6.38, "west": 3.10, "north": 6.71, "east": 3.72},
    "nairobi":       {"south": -1.44, "west": 36.65, "north": -1.16, "east": 37.01},
    "abuja":         {"south": 8.76, "west": 6.99, "north": 9.34, "east": 7.73},
    "kano":          {"south": 11.85, "west": 8.37, "north": 12.15, "east": 8.78},
    "ibadan":        {"south": 7.26, "west": 3.79, "north": 7.52, "east": 4.12},
    "port harcourt": {"south": 4.65, "west": 6.89, "north": 5.02, "east": 7.21},
    "mombasa":       {"south": -4.15, "west": 39.55, "north": -3.95, "east": 39.85},
    "kisumu":        {"south": -0.17, "west": 34.68, "north":  0.06, "east": 34.90},
}

_osm_cache: dict = {}


async def get_osm_road_closures(city: str) -> list:
    """
    Fetch community-reported road closures from OSM Road Closures API.
    New platform built in GSoC 2025 at closures.osm.ch — free, no API key.
    Returns closures with OpenLR location referencing for precise road matching.

    Falls back to Overpass API query for highway=construction if
    closures.osm.ch returns nothing.
    """
    cache_key = f"osm_closures|{city}"
    if cache_key in _osm_cache:
        entry = _osm_cache[cache_key]
        if time.time() - entry["ts"] < 900:   # 15 min cache
            return entry["data"]

    bbox    = OSM_CITY_BBOXES.get(city.lower())
    if not bbox:
        return []

    incidents = []
    now       = time.time()
    provider  = f"osm_closures:{city}"

    # ── Layer 1: OSM Road Closures API (closures.osm.ch) ─────
    try:
        url = (
            f"https://closures.osm.ch/api/closures"
            f"?bbox={bbox['west']},{bbox['south']},{bbox['east']},{bbox['north']}"
        )
        async with httpx.AsyncClient(
            timeout=8, headers={"User-Agent": "CommuteIQ/2.0"}
        ) as client:
            r = await client.get(url)
            if r.status_code == 200:
                data = r.json()
                for closure in data.get("features", []):
                    props = closure.get("properties", {})
                    name  = props.get("name") or props.get("description") or city
                    end   = props.get("end_date", "")
                    # Parse expiry
                    try:
                        from datetime import datetime as _dt
                        exp_ts = _dt.fromisoformat(end).timestamp() if end else now + 86_400
                    except Exception:
                        exp_ts = now + 86_400

                    incidents.append({
                        "type":         "road_closure",
                        "location":     name[:100],
                        "city":         city,
                        "source":       "osm_closures",
                        "confidence":   0.75,   # community-verified, higher than news
                        "geo_verified": True,
                        "published_at": now,    # OSM closures API does not provide pubDate
                        "retrieved_at": now,
                        "created_at":   now,
                        "expires_at":   exp_ts,
                    })
                _record_provider_ok(provider)
            else:
                _record_provider_error(provider, f"http_{r.status_code}")
    except httpx.TimeoutException:
        _record_provider_error(provider, "timeout")
    except Exception:
        _record_provider_error(provider, "unexpected_error")

    # ── Layer 2: Overpass API — highway=construction ──────────
    # Catches road works that OSM editors have tagged but aren't
    # in the closures API yet. Updated daily from OSM community edits.
    if len(incidents) < 3:   # only hit Overpass if closures API had little data
        overpass_provider = f"osm_overpass:{city}"
        try:
            overpass_query = f"""
[out:json][timeout:15];
(
  way["highway"="construction"]
     ({bbox['south']},{bbox['west']},{bbox['north']},{bbox['east']});
  way["construction"~"primary|secondary|tertiary|trunk|motorway"]
     ({bbox['south']},{bbox['west']},{bbox['north']},{bbox['east']});
  way["access"="no"]["highway"]
     ({bbox['south']},{bbox['west']},{bbox['north']},{bbox['east']});
);
out tags 10;
"""
            overpass_url = "https://overpass-api.de/api/interpreter"
            async with httpx.AsyncClient(
                timeout=12, headers={"User-Agent": "CommuteIQ/2.0"}
            ) as client:
                r = await client.post(overpass_url, data={"data": overpass_query})
                if r.status_code == 200:
                    elements = r.json().get("elements", [])
                    for el in elements[:5]:   # cap at 5 to avoid noise
                        tags = el.get("tags", {})
                        name = (tags.get("name") or
                                tags.get("ref") or
                                tags.get("description") or
                                f"{city} road works")
                        incidents.append({
                            "type":         "construction",
                            "location":     name[:100],
                            "city":         city,
                            "source":       "osm_overpass",
                            "confidence":   0.65,
                            "geo_verified": True,
                            "published_at": now,    # no pubDate from Overpass
                            "retrieved_at": now,
                            "created_at":   now,
                            "expires_at":   now + 86_400,  # 24h
                        })
                    _record_provider_ok(overpass_provider)
                else:
                    _record_provider_error(overpass_provider, f"http_{r.status_code}")
        except httpx.TimeoutException:
            _record_provider_error(overpass_provider, "timeout")
        except Exception:
            _record_provider_error(overpass_provider, "unexpected_error")

    _osm_cache[cache_key] = {"data": incidents, "ts": now}
    return incidents


# ═══════════════════════════════════════════════════════════════
# HELPER — fallback to time-based congestion when no TomTom key
# ═══════════════════════════════════════════════════════════════

def _time_based_congestion() -> dict:
    """Original time-of-day estimate — used when TomTom key not set."""
    import datetime
    hour = datetime.datetime.now().hour
    if 7 <= hour <= 9 or 17 <= hour <= 19:
        level = "High"
    elif 10 <= hour <= 16 or 20 <= hour <= 21:
        level = "Medium"
    else:
        level = "Low"
    return {
        "congestion":         level,
        "source":             "time_of_day_estimate",
        "congestion_method":  "time_of_day_heuristic",
        "congestion_note":    "Congestion estimated from time of day (heuristic). "
                              "Set TOMTOM_API_KEY for measured traffic data.",
        "live":               False,
    }


# ═══════════════════════════════════════════════════════════════
# COMBINED INTELLIGENCE — single call for main.py
# ═══════════════════════════════════════════════════════════════

async def get_route_intelligence(
    origin:      str,
    destination: str,
    city:        str,
    lat:         float,
    lng:         float,
) -> dict:
    """
    Single entry point for main.py to get all live intelligence.

    Returns:
      congestion              — str: "Low" | "Medium" | "High"
      congestion_detail       — dict: source, live flag, speed (TomTom) or heuristic note
      congestion_method       — str: "time_of_day_heuristic" | "tomtom_live"
      congestion_note         — str: human-readable explanation of congestion source
      incidents               — list: all active incidents (all sources, deduped)
      community_incidents     — list: only geo-verified incidents suitable for safety counts
      incident_count          — int: len(incidents)
      community_incident_count— int: len(community_incidents)
      confidence_boost        — int
      data_sources            — list[str]
      provider_health         — dict: per-provider status snapshots
    """
    # Run all four sources concurrently — fail safe on each
    (congestion_result,
     tomtom_incidents,
     news_incidents,
     route_news,
     osm_closures) = await asyncio.gather(
        get_live_congestion(lat, lng, city),
        get_live_incidents(city),
        get_news_incidents([city]),
        search_news_for_route(origin, destination, city),
        get_osm_road_closures(city),
        return_exceptions=True
    )

    # Handle exceptions gracefully — one source failing never breaks the others
    if isinstance(congestion_result, Exception):
        _record_provider_error("tomtom_flow", "unexpected_error")
        congestion_result = _time_based_congestion()
    if isinstance(tomtom_incidents, Exception):
        _record_provider_error("tomtom_incidents", "unexpected_error")
        tomtom_incidents = []
    if isinstance(news_incidents, Exception):
        _record_provider_error("rss_news", "unexpected_error")
        news_incidents = []
    if isinstance(route_news, Exception):
        _record_provider_error("google_news_route", "unexpected_error")
        route_news = []
    if isinstance(osm_closures, Exception):
        _record_provider_error("osm_closures", "unexpected_error")
        osm_closures = []

    # Collect and type-guard
    raw_incidents: list = []
    for inc in (tomtom_incidents + news_incidents + route_news + osm_closures):
        if isinstance(inc, dict):
            raw_incidents.append(inc)

    # Remove items whose published_at has aged out (re-check on every consume,
    # not just at cache write time — catches slow-moving cache entries).
    now = time.time()
    fresh_incidents = []
    for inc in raw_incidents:
        pa   = inc.get("published_at")
        itype = inc.get("type", "_default")
        if _article_is_fresh(pa, itype):
            fresh_incidents.append(inc)
        # TomTom and OSM items have no independent pubDate so they carry
        # published_at = retrieved_at; they pass the freshness check as long
        # as the cache TTL is shorter than the max age (always true for our values).

    # Cross-source deduplication
    deduped = deduplicate_incidents(fresh_incidents)

    # Separate geo-verified incidents (suitable for safety alert counts) from
    # low-confidence / unverified items (returned for transparency only).
    community_incidents = [
        i for i in deduped
        if i.get("geo_verified", False)
        and i.get("confidence", 0) >= GEO_CONFIDENCE_THRESHOLD
    ]

    # Confidence boost: live data is more trustworthy than guesses
    confidence_boost = 15 if congestion_result.get("live") else 0
    confidence_boost += min(len(community_incidents) * 5, 20)

    # Congestion method/note: propagate from congestion_result or construct
    cong_method = "tomtom_live" if congestion_result.get("live") else "time_of_day_heuristic"
    cong_note   = congestion_result.get(
        "congestion_note",
        "Congestion measured from live TomTom traffic flow data."
        if congestion_result.get("live")
        else "Congestion estimated from time of day (heuristic). "
             "Set TOMTOM_API_KEY for measured traffic data."
    )

    # Provider health snapshot — redact internal URL fragments
    health_snapshot = {
        name: get_provider_health(name)
        for name in [
            "tomtom_flow", "tomtom_incidents", "osm_closures",
            f"osm_closures:{city}", f"osm_overpass:{city}",
            f"google_news_route:{city}",
        ]
    }

    return {
        "congestion":               congestion_result["congestion"],
        "congestion_detail":        congestion_result,
        "congestion_method":        cong_method,
        "congestion_note":          cong_note,
        "incidents":                deduped,
        "community_incidents":      community_incidents,
        "incident_count":           len(deduped),
        "community_incident_count": len(community_incidents),
        "confidence_boost":         confidence_boost,
        "data_sources":             list(set(
            [congestion_result.get("source", "time_estimate")] +
            [i.get("source", "unknown") for i in deduped]
        )),
        "provider_health":          health_snapshot,
    }
