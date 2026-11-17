# CommuteIQ — Engineering Handoff Document

> **Purpose:** Complete transfer of development context to a new environment.
> Source of truth: repository code and git history as of commit `12e53cc`.
> Items marked **[UNVERIFIED]** could not be confirmed from code alone.

---

## 1. Architecture

### Repository layout

```
CommuteIQ-IBM-Bob/
├── backend/
│   ├── app/
│   │   ├── main.py              — FastAPI app, all endpoints, ML prediction logic
│   │   ├── live_intelligence.py — TomTom / RSS / Google News / OSM evidence layer
│   │   ├── routing.py           — OSRM routing wrapper
│   │   ├── storage.py           — Supabase / in-memory report storage
│   │   ├── weather_intelligence.py — Open-Meteo weather + flood risk
│   │   ├── transit_data.py      — matatu routes, walk-to-transit suggestions
│   │   ├── restriction_data.py  — legal mode restrictions by city (Nigeria + Kenya)
│   │   ├── models.py            — Pydantic request schemas (PredictRequest, ReportRequest)
│   │   ├── safety.py            — safety score helpers
│   │   ├── congestion.py        — congestion utilities
│   │   └── geocode.py           — geocoding utilities
│   ├── models/                  — trained pkl artifacts (DO NOT MODIFY)
│   │   ├── travel_time_model.pkl
│   │   ├── commute_quality_model.pkl
│   │   ├── safety_scores.pkl
│   │   ├── encoders.pkl
│   │   ├── road_quality.pkl
│   │   ├── transport_modes.pkl
│   │   └── reference_predictions.json  — numerical regression baseline
│   ├── tests/                   — pytest test suite (5 files, 376 tests)
│   ├── train_models.py          — model training script (do not run without approval)
│   ├── requirements.txt         — dependency pins
│   ├── Procfile                 — `uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}`
│   ├── railway.json             — Railway deployment config (Nixpacks, restart on failure)
│   ├── MODEL_PROVENANCE.md      — artifact version metadata and known gaps
│   └── .env.example             — required environment variable names
├── frontend/
│   └── index.html               — single-file frontend (Leaflet map, vanilla JS)
└── docs/
    └── bob-development-log.md   — session development log
```

### Backend (FastAPI)

**Entry point:** `backend/app/main.py` via `uvicorn app.main:app`

**Model loading:** All six pkl artifacts are loaded once at module import time
(`load_models()` at lines 77–93, called at module level). They are module-global
variables. **Never call `load_models()` a second time** — this is guarded by
Python's module cache as long as all imports use `from app.main import …`
(not `from main import …`).

**API endpoints:**

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Model status, supported cities, live intelligence config |
| POST | `/predict` | v1 prediction (no routing hard-fail; kept for compatibility) |
| POST | `/v2/predict` | Primary endpoint — raises 503 on routing failure |
| POST | `/recommend` | Departure time windows (4 offsets: now, +15, +30, +60 min) |
| POST | `/report` | Submit community report (v1 — no ethics checks) |
| POST | `/v2/report` | Submit community report (v2 — ethics checks, privacy anonymization) |
| GET | `/reports` | List active community reports |
| GET | `/modes` | Available transport modes for a city |

**v2/predict data flow:**
1. Validate city + mode → HTTP 400 if unknown
2. Geocode origin + destination via Nominatim (OSM) → HTTP 503 if routing fails
3. Route via OSRM public demo server → distance_km, route_geometry
4. Weather via Open-Meteo (free, no key) → weather label
5. Live intelligence (TomTom + RSS + Google News + OSM Overpass) → congestion, incidents
6. Community reports from Supabase (or memory) → supabase_count
7. Road quality lookup from pkl dict → rq_score, rq_available
8. XGBoost travel time prediction (≥50 km) or formula fallback (<50 km)
9. RandomForest commute quality classification
10. Safety score lookup from pkl dict
11. Confidence score, route confidence (star rating), departure advice, AI explanation
12. Return `PredictResponseV2`

**Transport modes:** 15 modes across Nigeria (danfo, BRT, okada, keke, rideshare,
driving, walking) and Kenya (matatu, bus, boda_boda, tuk_tuk, taxi, driving, walking).
All motorised modes use the OSRM `driving` profile as the physical road baseline;
mode-specific speed multipliers are applied from `transport_modes.pkl`.

**Supported cities (13):** Lagos, Abuja, Kano, Ibadan, Port Harcourt, Enugu, Asaba,
Benin City (Nigeria); Nairobi, Mombasa, Kisumu, Nakuru, Eldoret (Kenya).

### Frontend

Single HTML file: `frontend/index.html`. No build step, no npm, no framework.

**API base URL resolution** (priority order):
1. `<meta name="commuteiq-api-base" content="...">` — set this for any
   deployment where frontend and backend are on different domains
2. `window.location.origin` — when served from the same host as the API
3. `http://localhost:8000` — local development fallback

**Critical:** if the meta tag is empty and the frontend is on a different domain
than the API, all fetches silently fail (no console warning, no user message).
Always set the meta tag for split-host deployments.

### External dependencies (no API keys unless noted)

| Service | Purpose | Key required |
|---|---|---|
| Nominatim (OSM) | Geocoding (1 req/sec rate limit) | No |
| OSRM public demo | Road routing | No — but rate-limited; self-host for production |
| Open-Meteo | Weather + flood risk | No |
| TomTom Traffic Flow | Live congestion (measured speeds) | Yes — `TOMTOM_API_KEY` (free tier) |
| RSS feeds (8 feeds) | Nigerian + Kenyan news incident scanning | No |
| Google News | Targeted incident search | No |
| OSM Overpass API | Road closures + construction | No |
| Supabase | Community report storage | Yes — `SUPABASE_URL` + `SUPABASE_KEY` |

### Deployment

- **Platform:** Railway (confirmed from `railway.json`) or any platform that
  reads a Procfile
- **Build:** Nixpacks (no Dockerfile needed, though a `Dockerfile` exists)
- **Start command:** `uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}`
- **Restart policy:** ON_FAILURE, max 10 retries
- **Working directory:** `backend/` (Procfile is at `backend/Procfile`)
- **Frontend:** static file, deployable independently (Vercel, Netlify, or
  served from the same host)

---

## 2. Current development status

### Commit history (all on `origin/main`)

```
12e53cc  Model compatibility diagnostic: fix dual import path, add reference
         predictions, model provenance, 32 new tests          ← HEAD
fb50228  Phase 2.1 follow-up: dedup sentinel, geo labels, provenance schema,
         model date, outage flag, boundary test fix
538374a  Phase 2.1: evidence integrity, geographic relevance, frontend
         transparency, deduplication, provider health, incident provenance
40f6dfa  Phase 2: data integrity, provenance, sanity guards, and regression tests
c5e06d6  Fix safety score aliases: all 13 cities now have validated scores
c3fdc8a  Phase 1: XSS fix, Abuja flood coords, dir() guard, safety score transparency
fdbe982  Add frontend, safety/terminus fixes, and session development log
```

### Phase 1 (`c3fdc8a`) — Foundation fixes
- XSS sanitisation in frontend (`.textContent` not `.innerHTML` for user data)
- Abuja flood zone coordinates corrected
- `dir()` guard removed from geocoding fallback
- Safety score transparency: `available`, `is_hardcoded_override`, `granularity` fields

### Phase 2 (`40f6dfa`) — Data integrity and provenance
- Safety score aliases: all 13 cities now resolve (kisumu, eldoret, port harcourt, asaba, benin city)
- Travel time sanity guards: negative / NaN / implausible values fall through to formula
- Road quality sentinel: `road_quality_available = False` when score is the 50.0 fallback
- Supabase schema: `expires_at` column added (backward-compatible)
- v2/predict raises HTTP 503 on routing failure (instead of silently using 12 km fallback)
- Regression tests for all above

### Phase 2.1 (`538374a` + `fb50228`) — Evidence integrity
Six evidence-integrity areas implemented in `live_intelligence.py`:

| Area | What was done |
|---|---|
| Evidence freshness | `_parse_rfc2822`, `_article_is_fresh`, `MAX_ARTICLE_AGE_S` per incident type; cache refreshes do not reset original `published_at` |
| Geographic relevance | `_assign_city`, `CITY_TITLE_KEYWORDS` + `CITY_BODY_KEYWORDS` for all 13 cities; body-only matches excluded; Benin City bare `"benin"` keyword removed to prevent Benin Republic false matches |
| Frontend transparency | `road_quality_available` guard in JS (sentinel never shown as measurement); safety score provenance note dynamically built from `is_hardcoded`, `granularity`, `is_historical` |
| Deduplication | `_incident_fingerprint`, `deduplicate_incidents`, `corroborating_sources`; `_NO_DEDUP_SENTINEL` for items with no URL and no meaningful location (never merged) |
| Provider health | `_record_provider_ok/error`, `get_provider_health`, `error_category`; `all_providers_failed` flag distinguishes zero-incident from provider-down |
| Incident provenance | `community_reports` = Supabase only; `third_party_alerts` = TomTom/news/OSM; `total_incident_alerts` = sum; all three in API response and `PredictResponse` schema |

Follow-up corrections: `geo_verified` renamed to `geo_relevance_verified` +
`location_confirmed`; `source_category` added; `training_data_date` in response;
clock-drift boundary test fixed.

### Phase 2.1.1 (`12e53cc`) — Model compatibility diagnostic
- Dual import path bug fixed: `from main import app` → `from app.main import app` in
  `test_phase2_1_evidence_integrity.py` (eliminated one redundant `load_models()` call)
- `models/reference_predictions.json` created: 5 XGBoost + 5 RandomForest baseline predictions
- `MODEL_PROVENANCE.md` created: confirmed vs inferred artifact metadata
- `requirements.txt` corrected: sklearn pin `1.5.0 → 1.8.0`, XGBoost pin `2.0.3 → 3.4.1`
- 32 new tests in `test_model_compatibility.py`

---

## 3. Testing

### Run the full suite

```bash
cd backend
python -m pytest tests/ -q --tb=short
```

**Expected:** `376 passed, 0 failed, 6 warnings`

### Run a single file

```bash
python -m pytest tests/test_model_compatibility.py -v
python -m pytest tests/test_phase2_1_evidence_integrity.py -v
```

### Test files

| File | Tests | Covers |
|---|---|---|
| `test_phase1_fixes.py` | ~80 | Safety score aliases, mode multipliers, XSS guard |
| `test_phase2_fixes.py` | ~80 | Travel time sanity, road quality sentinel, 503 on routing failure |
| `test_safety_and_terminus.py` | ~40 | Safety score edge cases, matatu terminus restriction |
| `test_v2_predict_routing_failure.py` | ~15 | HTTP 503 on OSRM failure |
| `test_phase2_1_evidence_integrity.py` | 105 | All Phase 2.1 evidence integrity areas |
| `test_model_compatibility.py` | 32 | Import dedup, pkl version metadata, numerical reference, warning counts |

### Known warnings (all pre-existing, none new)

| Warning | Count per run | Cause | Risk |
|---|---|---|---|
| `InconsistentVersionWarning: DecisionTreeClassifier 1.8.0 → 1.9.1` | ~402 | 200 DT estimators in RF pkl loaded under sklearn 1.9.1 | Medium — versions confirmed different; reference tests detect output drift |
| `InconsistentVersionWarning: RandomForestClassifier 1.8.0 → 1.9.1` | ~2 | RF wrapper itself | Same as above |
| `UserWarning` (XGBoost) | 0–1 | Deprecated pickle format (`XGBoosterUnserializeFromBuffer`); NOT a version mismatch (XGBoost 3.4.1 == current) | Low — format warning, model loads correctly |
| `StarletteDeprecationWarning` | 1 | `starlette==1.9.0` prefers `httpx2` over `httpx`; `httpx2` not installed | Low — will become hard error in a future Starlette release |

### Numerical reference tests

`test_model_compatibility.py::TestTravelTimeReferenceOutputs` and
`TestQualityReferenceOutputs` compare raw model predictions against
`models/reference_predictions.json` (tolerance: 0.01 min / 0.01 probability).

A failure here means a dependency upgrade has silently shifted model outputs.
**Do not update the reference file without documenting why outputs changed.**

---

## 4. Environment

### Python version

**3.14.6** (confirmed from test runner). No lower bound tested.

### Install dependencies

```bash
cd backend
pip install -r requirements.txt
```

### Required environment variables

Set in `.env` (copy from `.env.example`), or as platform secrets:

| Variable | Purpose | Required for |
|---|---|---|
| `SUPABASE_URL` | Supabase project URL | Community report persistence; falls back to in-memory without it |
| `SUPABASE_KEY` | Supabase anon or service key | Same as above |
| `FRONTEND_URL` | Allowed CORS origin for the frontend | Production CORS; defaults to `*` (permissive) |
| `TOMTOM_API_KEY` | TomTom Traffic Flow API key | Live congestion data; falls back to time-of-day heuristic without it |
| `PORT` | Server port | Used by Procfile; defaults to 8000 |

**Never commit actual values.** The `.env` file is in `.gitignore`.

### Model artifacts

Located at `backend/models/*.pkl`. **Do not modify, delete, or overwrite any pkl file.**
If `train_models.py` is run, it only regenerates three artifacts
(`travel_time_model.pkl`, `safety_scores.pkl`, `encoders.pkl`) — the other three
(`commute_quality_model.pkl`, `road_quality.pkl`, `transport_modes.pkl`) are listed
as "pre-generated separately" and their training scripts are unknown (see `MODEL_PROVENANCE.md`).

Artifact version summary:

| Artifact | Type | Serialised with | Current env |
|---|---|---|---|
| `commute_quality_model.pkl` | `RandomForestClassifier` (200 trees, 8 features) | sklearn **1.8.0** (confirmed from pkl bytes) | sklearn 1.9.1 |
| `travel_time_model.pkl` | `XGBRegressor` (300 trees, 8 features) | XGBoost **3.4.1** (confirmed from `save_config`) | XGBoost 3.4.1 |
| `safety_scores.pkl` | `dict` (state/county crash scores) | Unknown | N/A |
| `encoders.pkl` | `dict` (feature maps, city/mode lookups) | Unknown | N/A |
| `road_quality.pkl` | `dict` (20,880 named road scores + city averages) | Unknown | N/A |
| `transport_modes.pkl` | `dict` (15 mode profiles) | Unknown | N/A |

### Frontend API configuration

Edit the meta tag in `frontend/index.html` line 13:

```html
<!-- For split-host deployment (frontend ≠ backend domain): -->
<meta name="commuteiq-api-base" content="https://your-backend.onrender.com">

<!-- For same-host or local dev: leave empty -->
<meta name="commuteiq-api-base" content="">
```

---

## 5. Phase 2.2 backlog — Reliability and User-Facing Evidence Transparency

These items were defined before the environment transition. No implementation
has started.

### P2.2-1 · Frontend provider outage banner (HIGH)
**Problem:** `live_intelligence.all_providers_failed` is in the API response but
the frontend does not read it. Users see zero incidents with no indication that
all data sources may be down.  
**Acceptance:** When `all_providers_failed == true`, show a yellow banner
("Live traffic data temporarily unavailable — estimates based on time of day").
**File:** `frontend/index.html`

### P2.2-2 · Frontend training date display (MEDIUM)
**Problem:** `training_data_date` is in the API response (`PredictResponse.training_data_date`)
but not rendered anywhere in the UI.  
**Acceptance:** Display "Model trained: YYYY-MM-DD" in the result footer alongside the
confidence score. Muted text, not a primary element.  
**File:** `frontend/index.html`

### P2.2-3 · Frontend incident provenance split (MEDIUM)
**Problem:** The incident panel shows a single count. The API now returns
`community_reports` (Supabase), `third_party_alerts` (TomTom/news/OSM), and
`total_incident_alerts` separately — but the frontend shows only one number.  
**Acceptance:** Show "N community reports + M live alerts" with a tooltip
explaining the sources. Never conflate the two.  
**File:** `frontend/index.html`

### P2.2-4 · Bounded deduplication for sentinel items (MEDIUM)
**Problem:** Items with no URL and no meaningful location (`_NO_DEDUP_SENTINEL`)
bypass deduplication and accumulate without bound if a provider re-fetches the
same structureless event (e.g. a persistent OSM construction zone re-fetched
every 10 minutes).  
**Acceptance:** Cap sentinel items at N per (city, type, 30-min window). Exact
value TBD.  
**File:** `backend/app/live_intelligence.py`

### P2.2-5 · Per-feed RSS health tracking (LOW)
**Problem:** `_record_provider_ok/error` is called once per provider category.
Individual RSS feed failures are swallowed if any feed in the group succeeds.  
**Acceptance:** Track per-feed health so `get_provider_health("rss_trafficradio961")`
can report individually.  
**File:** `backend/app/live_intelligence.py`

### P2.2-6 · API base configuration diagnostic (LOW)
**Problem:** A missing or empty `commuteiq-api-base` meta tag on a split-host
deployment produces silent fetch failures — no warning to the developer or user.  
**Acceptance:** On first failed fetch, check whether `API_BASE` matches
`window.location.origin`. If they differ and the meta tag is empty, log a
console warning: "CommuteIQ: API_BASE resolved to same origin but fetch failed.
Set <meta name='commuteiq-api-base'> for split-host deployments."  
**File:** `frontend/index.html`

### P2.2-7 · Safety score provenance visual differentiation (LOW)
**Problem:** The `#safety-note` element builds correct provenance text but
displays everything in the same plain style — a hardcoded-override warning looks
identical to a normal "state-level aggregate" note.  
**Acceptance:** Hardcoded-override warnings (`is_hardcoded == true`) display
in amber/orange (`var(--yellow)`) with a warning prefix. Normal provenance
remains muted.  
**File:** `frontend/index.html`

---

## 6. Phase 3 backlog

No Phase 3 implementation has started. These items were identified during
Phase 2 review.

### P3-1 · Long-distance motorcycle prediction reachability audit
**Problem:** XGBoost produces implausible speeds for motorcycle modes at ≥50 km
(boda_boda 101 km/h over 300 km; okada 85 km/h over 500 km). These are
currently assumed non-user-facing because urban trips are <50 km. This has
**not been verified** — OSRM can in principle return >50 km for inter-city
origin/destination pairs entered by users.  
**Task:** Check whether any city pair in `CITY_COORDS` can produce a route
≥50 km, and whether any user-reachable origin/destination combination across
the 13 cities exceeds the threshold. If yes, add a mode-specific distance cap
that falls through to the formula for motorcycle modes regardless of distance.

### P3-2 · Cross-version model numerical consistency
**Problem:** `reference_predictions.json` was recorded with sklearn 1.9.1 loading
a 1.8.0 artifact. Whether sklearn 1.8.0 produces identical floating-point
outputs has not been established.  
**Task:** Reproduce predictions in a clean sklearn 1.8.0 environment and compare
against the reference file. Document whether outputs match or diverge.

### P3-3 · OSRM public demo server reliability
**Problem:** `v2/predict` raises HTTP 503 when OSRM fails. The public demo
server (`router.project-osrm.org`) is not SLA-backed and is documented as
prototype-only.  
**Task:** Evaluate self-hosted OSRM or a paid routing alternative (Mapbox,
GraphHopper cloud) for production. Until then, 503 errors from OSRM are
user-visible and expected under load.

### P3-4 · Travel time model temporal forecasting limitation
**Problem:** The XGBoost model was trained on a static dataset with no
time-series component. It cannot forecast future congestion — it predicts
based on current conditions encoded at prediction time.  
**Task:** Document this explicitly in the API response (not currently stated).
Consider adding a `forecast_capable: false` field to `/health`.

### P3-5 · Routing distance vs travel time consistency
**Problem:** `predict_travel_time` uses OSRM road distance for ≥50 km trips
(XGBoost path) but the formula fallback uses a straight-line speed approximation.
For the same route, switching between paths mid-session could produce
discontinuous travel times.  
**Task:** Audit whether any route type can flip between paths across repeated
calls (e.g. if OSRM returns slightly different distances for the same query).

---

## 7. Known limitations

| # | Limitation | Status | Risk |
|---|---|---|---|
| L1 | sklearn 1.8.0 artifact loaded under 1.9.1 — numerical consistency unverified | Reference tests established (current env only); cross-version comparison not done | Medium |
| L2 | Long-distance motorcycle prediction reachability not confirmed | Assumed non-user-facing; not verified for inter-city inputs | Unknown — P3-1 |
| L3 | Benin City keyword `"benin"` removed to prevent Benin Republic false matches; `"benin city"` and `"edo state"` are the only safe keywords | Implemented and tested | Known gap in keyword coverage |
| L4 | RSS items without `<pubDate>` are excluded entirely (no timestamp = excluded) | No fallback to `retrieved_at` with reduced confidence | Low — conservative but discards valid incidents |
| L5 | TomTom / OSM items use `retrieved_at` as `published_at` (no event-time clock) | Bounded by cache TTL | Low |
| L6 | Provider health state is in-memory — lost on restart | Outage across a restart boundary is invisible | Low |
| L7 | `commuteiq-api-base` meta tag missing on split-host deployment causes silent fetch failures | No runtime warning exists | Medium for new deployments |
| L8 | `road_quality_available = False` for cities not covered by named road pkl | Sentinel 50.0 is never shown; "data unavailable" is shown instead | Resolved — guarded in frontend and backend |
| L9 | Nairobi and Lagos safety scores are manually set estimates (not formula-derived from crash data) | Flagged in API response (`is_hardcoded_override: true`) | Documented |
| L10 | OSRM public demo server has no SLA; `v2/predict` raises 503 when it is unavailable | Expected behaviour; documented | Known operational risk |
| L11 | `commute_quality_model.pkl` training script and training data are unknown | Documented in MODEL_PROVENANCE.md | Unresolved provenance gap |
| L12 | `httpx2` not installed; Starlette 1.9.0 warns on import of `TestClient` | Low priority; will escalate to hard error in future Starlette | Low |

---

## 8. Research considerations

CommuteIQ is a product and must be kept **strictly separate** from any academic
research track. The following research topics have been mentioned in the
development context but have no implementation in this repository:

- **SHAP explanation aggregation** — using SHAP values to explain individual
  predictions in a privacy-preserving aggregate form
- **Privacy leakage analysis** — assessing whether model explanations expose
  sensitive training data characteristics
- **Secure explanation queries** — cryptographic or differential-privacy
  mechanisms for querying model explanations

These are research questions, not engineering tasks. Any implementation of
these concepts in CommuteIQ would require:
1. Separate research codebase / branch
2. IRB or ethics review (if human data is involved)
3. Explicit separation from the production prediction pipeline

Do not add SHAP dependencies, differential privacy libraries, or explanation
aggregation code to `backend/app/` without a formal research design review.

---

## 9. Operational guidance

### Start the backend locally

```bash
cd backend
cp .env.example .env          # fill in real values
pip install -r requirements.txt
uvicorn app.main:app --reload  # dev mode with auto-reload
# API available at http://localhost:8000
```

### Start the frontend locally

No build step needed. Open `frontend/index.html` directly in a browser, or
serve it with any static file server:

```bash
# Python built-in server from repo root:
python -m http.server 3000 --directory frontend
# Then visit http://localhost:3000
```

The frontend defaults to `http://localhost:8000` for the API when running locally.

### Verify API health

```bash
curl http://localhost:8000/health
# Should return: {"status":"ok","version":"2.0.0","model_version":"v2.1",...}
```

Check model loading status in the `"models"` key — all five should be `true`.
Check `"live_intelligence.tomtom_active"` — will be `false` if `TOMTOM_API_KEY`
is not set (congestion will use time-of-day heuristic instead of measured data).

### Run tests

```bash
cd backend
python -m pytest tests/ -q --tb=short
# Expected: 376 passed, 0 failed, 6 warnings
```

### Avoid accidentally changing trained models

- **Never run** `python train_models.py` unless explicitly approved — it
  overwrites `travel_time_model.pkl`, `safety_scores.pkl`, and `encoders.pkl`
- **Never run** `joblib.dump(...)` or `pickle.dump(...)` targeting any file
  in `backend/models/`
- The `commute_quality_model.pkl`, `road_quality.pkl`, and `transport_modes.pkl`
  files have no known regeneration script — if they are deleted, they cannot
  be recovered without retraining

### Emergency: check whether models loaded

```bash
curl http://localhost:8000/health | python -m json.tool | grep -A6 '"models"'
```

All five model keys should be `true`. A `false` value means a pkl file is
missing or unreadable — the API will fall back to rule-based defaults
(which produce less accurate but non-crashing predictions).

---

## 10. Next recommended task: Phase 2.2

### Implementation checklist (prioritized)

- [ ] **P2.2-1 (HIGH)** — Read `live_intelligence.all_providers_failed` in
  `frontend/index.html`. When `true`, show a yellow banner between the result
  header and the prediction card. No backend change needed.

- [ ] **P2.2-3 (MEDIUM)** — Split the incident display in `frontend/index.html`
  into `community_reports` + `third_party_alerts`. Both fields are already in
  the API response. No backend change needed.

- [ ] **P2.2-2 (MEDIUM)** — Display `training_data_date` in the result footer.
  The field (`MODEL_TRAINED = "2026-07-23"`) is already in the API response
  as `training_data_date`. No backend change needed.

- [ ] **P2.2-6 (LOW)** — Add a `console.warn` in `frontend/index.html` when
  `API_BASE` resolves to same origin but a fetch returns a network error
  and the meta tag is empty.

- [ ] **P2.2-4 (MEDIUM)** — Cap `_NO_DEDUP_SENTINEL` accumulation in
  `backend/app/live_intelligence.py`. Requires a per-(city, type, window) count
  inside `deduplicate_incidents`.

- [ ] **P2.2-7 (LOW)** — Apply amber styling to `#safety-note` when
  `safety_score_is_hardcoded == true`. CSS variable `--yellow` is already
  defined in `index.html`.

- [ ] **P2.2-5 (LOW)** — Add per-feed health tracking in
  `backend/app/live_intelligence.py` for each RSS feed URL.

### Acceptance criteria for Phase 2.2 completion

1. `all_providers_failed` banner visible in UI when all providers return errors
   in a test scenario (can be triggered by clearing `TOMTOM_API_KEY` and
   blocking DNS for news feeds in a local test)
2. Incident panel displays community and third-party counts separately with
   correct labels; never shows a combined unlabelled total
3. `training_data_date` visible in result footer
4. `_NO_DEDUP_SENTINEL` accumulation bounded — a provider returning the same
   no-URL event 20 times in one session should not produce 20 identical items
5. No new test failures; full suite remains ≥ 376 passed
6. No new warnings added
7. All frontend changes are plain JS/HTML — no new libraries, no build step

---

*Document generated from repository state at commit `12e53cc` on 2026-11-17.*
*Branch: `main`. Remote: `origin/main` at `https://github.com/FaithOdhe-bot/CommuteIQ-IBM-Bob`.*
