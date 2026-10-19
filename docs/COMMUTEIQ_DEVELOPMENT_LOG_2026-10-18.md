# CommuteIQ — Development Log
**Session date:** 2026-10-18  
**Branch:** `main`  
**AI development assistant:** IBM Bob  
**Base commit:** `8d14cde Fix v2 routing failures and add regression tests`

---

## Overview

This session continued work started in an earlier session (Improvement 1, routing fix)
and delivered six further improvements plus a complete frontend. All changes are confirmed
by the file state in the repository; no speculative claims are made.

---

## Prior state going into this session

| Item | Status |
|------|--------|
| `/v2/predict` unbound-variable routing failure | Fixed in `8d14cde` (Improvement 1) |
| `backend/tests/test_v2_predict_routing_failure.py` | Committed in `8d14cde` |
| Frontend | Did not exist |
| `docs/bob-development-log.md` | Covered Improvement 1 only |

Uncommitted at session start: `backend/app/main.py` (87 ins / 36 del vs HEAD).

---

## Architecture (confirmed from code)

### Backend
- **Framework:** FastAPI 2.0.0 (`app = FastAPI(title="SmartCommute AI", version="2.0.0")`)
- **Endpoints:** `GET /health`, `POST /predict`, `POST /v2/predict`, `POST /recommend`,
  `GET /modes`, `POST /report`, `POST /v2/report`, `GET /reports`
- **Models:** XGBoost (`travel_time_model.pkl`) · RandomForest (`commute_quality_model.pkl`) ·
  `safety_scores.pkl` · `encoders.pkl` · `road_quality.pkl` · `transport_modes.pkl`
- **External APIs:** Nominatim (geocoding) · OSRM (routing) · Open-Meteo (weather) ·
  TomTom (optional live traffic, requires `TOMTOM_API_KEY` env var)
- **Cities:** 13 — Lagos, Abuja, Kano, Ibadan, Port Harcourt, Enugu, Asaba, Benin City
  (Nigeria); Nairobi, Mombasa, Kisumu, Nakuru, Eldoret (Kenya)

### Frontend
- Single-file vanilla HTML/CSS/JS (`frontend/index.html`, ~1 000 lines)
- Leaflet 1.9.4 for the route map
- Country → city two-tier selector
- No build step or framework

---

## Improvement 2 — Departure advice contradicting safety score (P2)

### Problem (confirmed bug)
`get_departure_advice()` returned `"Leave now — conditions are good."` regardless of the
`safety_score` returned by the ML model. A route scoring 26/100 (e.g. okada in a high-risk
corridor) could receive the "conditions are good" message when traffic was light and weather
was clear — a direct contradiction in the same response.

### Root cause
The function had no `safety_score` parameter; the score was computed elsewhere but never
threaded into the advice logic.

### Fix applied (in `backend/app/main.py`)
- Added `safety_score: float = 100.0` parameter (default preserves backward compatibility).
- Inserted a `LOW_SAFETY` branch (`safety_score < 50`) that fires after the weather branch
  but before the "conditions are good" fallback.
- Low-safety advice: `"Leave now, but exercise extra caution — safety score N/100 indicates elevated risk on this route."`
- All existing callers updated to pass `safety_score`.

### Tests added (`backend/tests/test_safety_and_terminus.py`)
`TestDepartureAdviceSafetyConsistency` — 10 tests covering:
- Low safety + clear/low-congestion does **not** say "conditions are good"
- Low safety advice acknowledges risk, includes score value, starts with "Leave now"
- High safety preserves original "conditions are good" phrasing
- Boundary: score = 50 is **not** low-safety; score = 49 **is**
- Rainy-weather branch fires before low-safety check
- Legacy callers without `safety_score` arg get default 100 (no regression)

---

## Improvement 3 — Matatu terminus warning on every Nairobi trip (P3)

### Problem (confirmed bug)
`suggest_alternative_mode()` triggered the `TERMINUS_RESTRICTION` matatu notice for every
Nairobi trip — including trips to Westlands, Karen, Kasarani and other outer suburbs. The
restriction (matatues cannot enter Nairobi CBD) is only relevant when the destination is
actually the city centre.

### Root cause
The `TERMINUS_RESTRICTION` branch had no destination check; it emitted the warning
unconditionally.

### Fix applied
- New helper `_dest_is_cbd(destination: str, city: str) -> bool`:
  1. Matches a small, high-confidence set of CBD tokens (`cbd`, `city centre`, `town`,
     `kenyatta avenue`, …).
  2. If the destination matches a known outer suburb in `NAIROBI_SUBURB_COORDS`, returns
     `False`.
  3. Unknown destinations default to `True` (conservative — better to show the notice than
     to silently omit it for a genuinely central destination).
- `suggest_alternative_mode()` now passes `destination` to `_dest_is_cbd()` before emitting
  the `TERMINUS_RESTRICTION` message.
- `destination: str = ""` added as a parameter to `suggest_alternative_mode()`.

### Tests added (`backend/tests/test_safety_and_terminus.py`)
`TestDestIsCbd` (13 tests) and `TestMatatuTerminusRestriction` (5 tests):
- Known CBD strings return `True`; known outer suburbs return `False`
- Enriched place strings like `"Kingeero, Nairobi"` resolve correctly
- Unknown destinations default to `True`
- CBD-bound matatu trips → terminus warning; suburb-bound → no warning
- Integration tests skip gracefully when pkl models are absent (CI without pkl)

---

## Improvement 4 — Leaflet map blank on load

### Problem (confirmed bug)
`frontend/index.html` rendered a blank grey rectangle instead of a map. Two independent
causes were identified.

### Root cause A — SRI hash mismatch (primary)
The `<script>` tag for `leaflet.js` from `unpkg.com` had an `integrity` attribute with a
stale SHA-512 hash (`/XN/WPeM=` suffix) that did not match the current CDN-served file
(`1lvTlZBo=` suffix). Browsers refused to execute the script under SRI enforcement.

### Root cause B — CSS `overflow: hidden` clipping tile panes
The `#map` element had `overflow: hidden` in its CSS rule, which clipped Leaflet's internal
tile pan containers and produced a blank canvas even when the JS loaded correctly.

### Fix applied
- Corrected the SRI hash to match `leaflet.js` v1.9.4 served by unpkg.
- Removed `overflow: hidden` from `#map`; wrapped the map in a `<div id="map-wrapper">` so
  the containing block constrains height without clipping tile panes.
- Added `typeof L === "undefined"` guard before `L.map()` initialization to prevent an
  uncaught `ReferenceError` if the CDN is unreachable.

---

## Improvement 5 — Frontend built from scratch

### What was created
`frontend/index.html` — a complete single-file frontend (~1 000 lines of HTML/CSS/JS):

| Feature | Detail |
|---------|--------|
| Country → city selector | Kenya / Nigeria → 13 cities; dynamically populated |
| Mode selector | Per-country modes loaded from `/modes` |
| Predict form | Calls `/v2/predict`, renders all response fields |
| Safety card | Score bar, colour-coded band |
| Commute quality card | Emoji + label + score |
| Confidence card | Score, label, contributors list |
| Route confidence | Star rating |
| Departure advice | Blue banner |
| Alternative suggestion | Orange banner |
| AI explanation | Paragraph |
| Leaflet map | Route polyline, origin/destination markers |
| Community reports | Incident chips with `(unverified)` label |
| Staggered departure | Shown only when `staggered_departure` is non-null |
| Weather / flood risk | Shown only when non-trivial |

---

## Improvement 6 — UI copy accuracy fixes

Three copy issues corrected in `generate_ai_explanation()` and the confidence module:

| Issue | Before | After |
|-------|--------|-------|
| Community reports verb | `"N reports confirmed"` | `"N reports filed"` |
| Incident chip label | none | `(unverified)` appended |
| Confidence "High" threshold | 85 | 90 — prevents over-claiming without live data |
| Confidence footer label | "Prediction confidence" | "Heuristic estimate" |
| Metric separation | unlabelled | Safety and commute quality labelled separately |

### Rationale
Reports are user-submitted and are not independently verified. "Confirmed" implied external
validation that does not exist. "Filed" is accurate. The confidence threshold change prevents
a clear-weather Saturday with 0 community reports from reaching "High confidence" — a score
built entirely from pattern matching (time-of-day + weather API) does not warrant that label
without live traffic data.

---

## Improvement 7 — Duplicate text in AI explanation removed

### Problem (confirmed bug)
`generate_ai_explanation()` assembled a paragraph that included:
1. The safety score value (also displayed as a dedicated safety card).
2. The `alt_suggestion` text (also displayed as an orange alternative-mode banner).

On every response these two pieces appeared twice — once in the paragraph, once as a card.

### Fix applied
Both fields are explicitly omitted from `generate_ai_explanation()`. A code comment marks
the omission intentional:

```python
# NOTE: safety score, mode rain warning, and alt_suggestion are intentionally
# omitted here — they are rendered as dedicated cards in the frontend and
# appending them here would cause every piece to display twice.
```

The `departure_advice` is also only appended to the paragraph when it contains a `"Wait"`
recommendation — the `"Leave now"` variants are already shown in the blue banner.

---

## Test results

| File | Tests | Passed | Failed |
|------|-------|--------|--------|
| `test_v2_predict_routing_failure.py` | 10 | 10 | 0 |
| `test_safety_and_terminus.py` | 31 | 31 | 0 |
| **Total** | **41** | **41** | **0** |

All tests run from `backend/` with `pytest tests/`.

---

## Pre-existing concerns (not modified this session)

These were identified during analysis and documented for the next session.

| Concern | Classification | Detail |
|---------|---------------|--------|
| Model version mismatch | **Possible concern** | `travel_time_model.pkl` and `commute_quality_model.pkl` were serialized with a different sklearn/XGBoost version than the runtime. joblib emits warnings; predictions still run via the formula fallback for urban routes (< 50 km). |
| `/predict` v1 routing fallback | **Confirmed divergence** | The v1 `/predict` endpoint silently falls back to 12.0 km on any routing exception (line 894). `/v2/predict` raises HTTP 503. The two endpoints have different failure contracts. |
| Dead code in `safety.py` | **Possible concern** | `safety.py` contains a `calculate_safety_score()` function that is never called from `main.py`; `get_safety_score()` in `main.py` re-implements the same logic. No harm, but creates a maintenance hazard. |
| Dead code in `congestion.py` | **Possible concern** | `congestion.py` is imported by no module; `estimate_congestion()` is defined inline in `main.py`. |
| Geocode fallback for non-Nairobi/non-Lagos cities | **Possible concern** | `geocode_place()` falls through to `CITY_COORDS[city]` (the city-centre coordinate) for cities like Mombasa, Kano, Kisumu if Nominatim fails. This produces a 0 km route. |
| `hash(user_id)` for staggered departure | **Possible concern** | Python's built-in `hash()` is randomized across processes (PYTHONHASHSEED). Two server restarts assign different slots to the same user. Low severity — slots are approximate suggestions, not commitments. |
| CORS `allow_origin_regex=".*"` | **Possible concern** | Combined with `allow_origins=["*"]`, this is fully open. Acceptable for a hackathon but should be tightened before public deployment. |

---

## Files changed this session

| File | Change |
|------|--------|
| `backend/app/main.py` | Modified — see Improvements 2, 3, 6, 7 above |
| `backend/tests/test_safety_and_terminus.py` | Created — 31 tests for P2 + P3 |
| `frontend/index.html` | Created — full single-file frontend |
| `docs/COMMUTEIQ_DEVELOPMENT_LOG_2026-10-18.md` | Created — this file |

Files **not** modified: `routing.py`, `safety.py`, `congestion.py`, `restriction_data.py`,
`storage.py`, `live_intelligence.py`, `weather_intelligence.py`, `transit_data.py`,
`models.py`, `requirements.txt`, all `*.pkl` model files.

---

## Next steps (recommended)

1. **Commit all uncommitted changes** — `main.py`, `test_safety_and_terminus.py`,
   `frontend/index.html`, this log.
2. **Retrain models** with the installed sklearn and XGBoost versions to clear the
   serialization warnings.
3. **Align v1 `/predict` failure behaviour** with v2 (HTTP 503 instead of silent 12 km
   fallback), or remove v1 if it is no longer needed.
4. **Geocode fallback for non-Nairobi non-Lagos cities** — add suburb coordinate tables for
   Mombasa, Kano, Port Harcourt, or at minimum detect and report the city-centre fallback.
5. **Deploy to Render or Railway** — not attempted this session.
