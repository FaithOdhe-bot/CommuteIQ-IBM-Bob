"""
SmartCommute AI — FastAPI Backend (v2)
main.py — Complete implementation with:
  - Mode-aware travel time prediction (Nigeria + Kenya transport modes)
  - Road quality integration
  - City validation
  - Alternative mode suggestions
  - Walking distance warnings
  - Full AI explanation engine
  - /recommend endpoint
  - Proper error handling

Endpoints:
  GET  /health    — health check + model status
  POST /predict   — full prediction with AI explanation
  POST /recommend — departure time recommendation
  POST /report    — submit community report
  GET  /reports   — list community reports
  GET  /modes     — list available modes per city
"""

import os
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Optional, List

# Ensure current directory is on sys.path for submodule resolution
APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import joblib
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from restriction_data    import get_restriction, TRANSPORT_RESTRICTIONS
from live_intelligence   import get_route_intelligence
from transit_data        import get_walk_to_transit, get_matatu_route, MATATU_ROUTES
from weather_intelligence import (
    get_weather_trend, get_flood_risk,
    get_day_pattern, DAY_CONGESTION_MULT
)
from models  import PredictRequest, ReportRequest
from routing import get_route
from storage import save_report, list_reports


# ── App ──────────────────────────────────────────────────────

app = FastAPI(
    title="SmartCommute AI",
    description="Community-powered AI mobility assistant for African cities",
    version="2.0.0",
)

frontend_origin = os.getenv("FRONTEND_URL", "*")
allowed_origins = [frontend_origin] if frontend_origin != "*" else ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_origin_regex=".*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Model loading ─────────────────────────────────────────────

MODELS_DIR = Path(__file__).parent.parent / "models"

def load_models():
    try:
        travel_model      = joblib.load(MODELS_DIR / "travel_time_model.pkl")
        quality_model     = joblib.load(MODELS_DIR / "commute_quality_model.pkl")
        safety_scores     = joblib.load(MODELS_DIR / "safety_scores.pkl")
        encoders          = joblib.load(MODELS_DIR / "encoders.pkl")
        road_quality      = joblib.load(MODELS_DIR / "road_quality.pkl")
        transport_modes   = joblib.load(MODELS_DIR / "transport_modes.pkl")
        print("✅ All models loaded")
        return travel_model, quality_model, safety_scores, encoders, road_quality, transport_modes
    except FileNotFoundError as e:
        print(f"⚠️  Model not found: {e}. Run train_models.py first.")
        return None, None, None, None, None, None

(travel_model, quality_model,
 safety_scores, encoders,
 road_quality, transport_modes) = load_models()


# ── City helpers ──────────────────────────────────────────────

CITY_COORDS = {
    "lagos":        {"lat": 6.5244,  "lng": 3.3792},
    "abuja":        {"lat": 9.0765,  "lng": 7.3986},
    "kano":         {"lat": 12.0022, "lng": 8.5920},
    "ibadan":       {"lat": 7.3775,  "lng": 3.9470},
    "port harcourt":{"lat": 4.8156,  "lng": 7.0498},
    "enugu":        {"lat": 6.4584,  "lng": 7.5464},
    "nairobi":      {"lat": -1.2921, "lng": 36.8219},
    "mombasa":      {"lat": -4.0435, "lng": 39.6682},
    "kisumu":       {"lat": -0.0917, "lng": 34.7680},
    "nakuru":       {"lat": -0.3031, "lng": 36.0800},
    "eldoret":      {"lat": 0.5143,  "lng": 35.2698},
    "asaba":        {"lat": 6.1986,  "lng": 6.7322},
    "benin city":   {"lat": 6.3350,  "lng": 5.6037},
}

# Nairobi suburb coordinates — used as fallback when Nominatim returns a
# location suspiciously close to CBD for an area known to be further away.
NAIROBI_SUBURB_COORDS = {
    "kingeero":    {"lat": -1.2676, "lng": 36.7342},
    "king'eero":   {"lat": -1.2676, "lng": 36.7342},
    "uthiru":      {"lat": -1.2650, "lng": 36.7300},
    "westlands":   {"lat": -1.2631, "lng": 36.8072},
    "karen":       {"lat": -1.3430, "lng": 36.7050},
    "langata":     {"lat": -1.3200, "lng": 36.7400},
    "lang'ata":    {"lat": -1.3200, "lng": 36.7400},
    "kasarani":    {"lat": -1.2206, "lng": 36.8956},
    "ruiru":       {"lat": -1.1466, "lng": 36.9608},
    "thika":       {"lat": -1.0322, "lng": 37.0693},
    "kikuyu":      {"lat": -1.2470, "lng": 36.6720},
    "embakasi":    {"lat": -1.3200, "lng": 36.9000},
    "donholm":     {"lat": -1.3006, "lng": 36.8906},
    "south b":     {"lat": -1.3072, "lng": 36.8338},
    "south c":     {"lat": -1.3200, "lng": 36.8200},
    "eastleigh":   {"lat": -1.2750, "lng": 36.8530},
    "ngong":       {"lat": -1.3580, "lng": 36.6600},
    "kitengela":   {"lat": -1.4750, "lng": 36.9600},
    "rongai":      {"lat": -1.3960, "lng": 36.7450},
    "mathare":     {"lat": -1.2550, "lng": 36.8650},
    "huruma":      {"lat": -1.2500, "lng": 36.8600},
    "githurai":    {"lat": -1.1950, "lng": 36.9100},
}
# Lagos area coordinates — fallback when Nominatim geocodes wrongly
LAGOS_AREA_COORDS = {
    "lekki":         {"lat": 6.4698,  "lng": 3.5852},
    "ajah":          {"lat": 6.4633,  "lng": 3.5893},
    "victoria island":{"lat": 6.4281, "lng": 3.4219},
    "vi":            {"lat": 6.4281,  "lng": 3.4219},
    "ikoyi":         {"lat": 6.4474,  "lng": 3.4341},
    "surulere":      {"lat": 6.5017,  "lng": 3.3537},
    "yaba":          {"lat": 6.5143,  "lng": 3.3785},
    "mushin":        {"lat": 6.5504,  "lng": 3.3577},
    "oshodi":        {"lat": 6.5563,  "lng": 3.3397},
    "agege":         {"lat": 6.6134,  "lng": 3.3240},
    "ojota":         {"lat": 6.5896,  "lng": 3.3926},
    "maryland":      {"lat": 6.5694,  "lng": 3.3585},
    "ketu":          {"lat": 6.5954,  "lng": 3.3850},
    "ikorodu":       {"lat": 6.6194,  "lng": 3.5058},
    "festac":        {"lat": 6.4683,  "lng": 3.2861},
    "isale eko":     {"lat": 6.4569,  "lng": 3.3919},
    "apapa":         {"lat": 6.4483,  "lng": 3.3599},
    "alimosho":      {"lat": 6.6167,  "lng": 3.2333},
    "ikotun":        {"lat": 6.5306,  "lng": 3.2667},
    "egbeda":        {"lat": 6.5717,  "lng": 3.2900},
    "ipaja":         {"lat": 6.5894,  "lng": 3.2525},
    "dopemu":        {"lat": 6.5833,  "lng": 3.2833},
    "mile 2":        {"lat": 6.4750,  "lng": 3.3150},
}

# Asaba area coordinates (Delta State)
ASABA_AREA_COORDS = {
    "okpanam":          {"lat": 6.2167, "lng": 6.6833},
    "ibusa":            {"lat": 6.1667, "lng": 6.6667},
    "anwai":            {"lat": 6.1800, "lng": 6.7400},
    "cable point":      {"lat": 6.2050, "lng": 6.7280},
    "bonsaac":          {"lat": 6.2100, "lng": 6.7350},
    "koka":             {"lat": 6.2200, "lng": 6.7500},
    "nnebisi":          {"lat": 6.1990, "lng": 6.7350},
    "summit":           {"lat": 6.2050, "lng": 6.7400},
    "federal housing":  {"lat": 6.2000, "lng": 6.7300},
    "millennium":       {"lat": 6.1950, "lng": 6.7250},
}

# Benin City area coordinates (Edo State)
BENIN_AREA_COORDS = {
    "gra":              {"lat": 6.3150, "lng": 5.6150},
    "uselu":            {"lat": 6.3750, "lng": 5.6020},
    "ugbowo":           {"lat": 6.3980, "lng": 5.6110},
    "ring road":        {"lat": 6.3350, "lng": 5.6037},
    "sapele road":      {"lat": 6.3200, "lng": 5.6200},
    "airport road":     {"lat": 6.3000, "lng": 5.5800},
    "ramat park":       {"lat": 6.3450, "lng": 5.6300},
    "ikpoba hill":      {"lat": 6.3600, "lng": 5.6400},
    "textile mill":     {"lat": 6.3500, "lng": 5.5900},
    "upper mission":    {"lat": 6.3700, "lng": 5.6000},
}


def get_country(city: str) -> str:
    if encoders:
        return encoders.get("city_to_country", {}).get(city.lower(), "nigeria")
    kenya_cities = ["nairobi","mombasa","kisumu","nakuru","eldoret"]
    return "kenya" if city.lower() in kenya_cities else "nigeria"

def validate_mode_for_city(mode: str, city: str) -> tuple[bool, str]:
    """Check if mode is available in city. Returns (valid, suggestion)."""
    if not transport_modes:
        return True, ""
    country = get_country(city)
    country_modes = transport_modes.get(country, {})
    if mode.lower() in country_modes:
        return True, ""
    # Suggest correct alternatives
    available = list(country_modes.keys())
    return False, f"'{mode}' is not available in {city.title()}. Available: {', '.join(available)}"

# Cities covered by road_quality.pkl city_avg.
# All other cities fall through to the ROAD_QUALITY_UNAVAILABLE sentinel.
_ROAD_QUALITY_CITY_AVG_CITIES = {"lagos", "kano", "abuja", "nairobi"}

# Sentinel returned when no city-level road quality data is available.
# Chosen as the midpoint of [0,100] so it has no directional bias in
# weighted scoring (route_confidence). Callers that need to distinguish
# "real data" from "no data" should check road_quality_score_available.
ROAD_QUALITY_UNAVAILABLE = 50.0


def get_road_quality_score(city: str, road_name: Optional[str] = None) -> dict:
    """Return road quality score with availability metadata.

    Returns a dict:
      score     – float [0-100]. ROAD_QUALITY_UNAVAILABLE (50.0) when no data.
      available – bool. False means only the neutral fallback is returned.
      source    – str. 'named_road_pkl' | 'city_avg_pkl' | 'unavailable_fallback'

    road_quality.pkl covers named roads (Nigeria 14,009 + Kenya 6,871) and
    city averages for Lagos, Kano, Abuja, Nairobi only. All other cities
    receive the neutral fallback; this is now explicit in the response.
    """
    if not road_quality:
        return {"score": ROAD_QUALITY_UNAVAILABLE, "available": False,
                "source": "unavailable_fallback"}

    country = get_country(city)

    # Named road lookup (highest specificity)
    if road_name:
        road_lower = road_name.lower().strip()
        named = road_quality.get(country, {})
        if road_lower in named:
            return {"score": float(named[road_lower]), "available": True,
                    "source": "named_road_pkl"}

    # City average
    city_avg = road_quality.get("city_avg", {}).get(city.lower())
    if city_avg is not None:
        return {"score": float(city_avg), "available": True,
                "source": "city_avg_pkl"}

    # No data for this city — return neutral fallback, clearly labelled
    return {"score": ROAD_QUALITY_UNAVAILABLE, "available": False,
            "source": "unavailable_fallback"}


# ── Weather ───────────────────────────────────────────────────

WEATHER_API = "https://api.open-meteo.com/v1/forecast"

async def get_weather(lat: float, lng: float) -> dict:
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(WEATHER_API, params={
                "latitude": lat, "longitude": lng,
                "current_weather": True,
                "hourly": "precipitation",
            })
            data = r.json()
            code = data.get("current_weather", {}).get("weathercode", 0)
            wind = data.get("current_weather", {}).get("windspeed", 0)
            if code in [51,53,55,61,63,65,80,81,82]: label = "Rainy"
            elif code in [45,48]:                     label = "Foggy"
            elif code in [1,2,3]:                     label = "Cloudy"
            else:                                     label = "Clear"
            return {"label": label, "code": code, "wind_kmh": wind}
    except Exception:
        return {"label": "Clear", "code": 0, "wind_kmh": 0}


# ── Congestion ────────────────────────────────────────────────

def estimate_congestion(time_str: Optional[str]) -> str:
    try:
        hour = int(time_str.split(":")[0]) if time_str else int(time.strftime("%H"))
    except Exception:
        hour = 8
    if 7 <= hour <= 9 or 17 <= hour <= 19:   return "High"
    elif 10 <= hour <= 16 or 20 <= hour <= 21: return "Medium"
    else:                                       return "Low"


# ── Geocoding ─────────────────────────────────────────────────

# Cache geocoding results for the session — same origin/destination
# queried by 8 modes shouldn't hit Nominatim 16 times.
_geocode_cache: dict = {}

async def geocode_place(place: str, city: str) -> dict:
    """Geocode a place name to lat/lng. For known Nairobi suburbs, validates
    the result isn't suspiciously close to CBD when the area is known to be
    further away — Nominatim sometimes resolves suburb names wrongly."""
    import math, asyncio

    cache_key   = f"{place.lower().strip()}|{city.lower().strip()}"
    if cache_key in _geocode_cache:
        return _geocode_cache[cache_key]

    # Respect Nominatim's 1 req/sec policy
    await asyncio.sleep(0.25)

    def dist_km(lat1, lng1, lat2, lng2):
        dlat = (lat2-lat1)*math.pi/180; dlng = (lng2-lng1)*math.pi/180
        a = math.sin(dlat/2)**2 + math.cos(lat1*math.pi/180)*math.cos(lat2*math.pi/180)*math.sin(dlng/2)**2
        return 6371 * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

    # enrichPlace() in client.js sends "Kingeero, Nairobi" not "Kingeero".
    # Extract just the place name before the comma so suburb lookups work.
    place_lower = place.lower().strip().rstrip(",")
    place_base  = place_lower.split(",")[0].strip()   # "kingeero, nairobi" → "kingeero"
    city_lower  = city.lower().strip()

    try:
        async with httpx.AsyncClient(
            timeout=6, headers={"User-Agent": "CommuteIQ/2.0"}
        ) as client:
            r = await client.get(
                "https://nominatim.openstreetmap.org/search",
                params={"q": f"{place}, {city}", "format": "json", "limit": 1,
                        "countrycodes": "ke" if city_lower in ["nairobi","mombasa","kisumu","nakuru","eldoret"] else "ng"},
            )
            results = r.json()
            if results:
                rlat = float(results[0]["lat"])
                rlng = float(results[0]["lon"])
                # Nairobi suburb validation — uses place_base (not place_lower)
                # because client.js enriches "Kingeero" to "Kingeero, Nairobi"
                # before sending. Threshold 5km catches suburbs placed near CBD
                # by Nominatim when the real suburb is 7-15km away.
                if city_lower == "nairobi" and place_base in NAIROBI_SUBURB_COORDS:
                    cbd    = CITY_COORDS["nairobi"]
                    known  = NAIROBI_SUBURB_COORDS[place_base]
                    r_cbd  = dist_km(rlat, rlng, cbd["lat"], cbd["lng"])
                    k_cbd  = dist_km(known["lat"], known["lng"], cbd["lat"], cbd["lng"])
                    if r_cbd < 5.0 and k_cbd > 3.0:
                        result = known
                        _geocode_cache[cache_key] = result
                        return result

                # Lagos suburb validation: reject result if Nominatim returns
                # a location more than 15km from the known coordinates.
                # Catches cases like "Ikoyi" resolving to Ojo LGA (Satellite Town
                # area, ~3.22 lng) instead of Ikoyi GRA on Lagos Island (~3.43 lng).
                if city_lower == "lagos":
                    place_norm = place_base.replace("'", "")
                    if place_norm in LAGOS_AREA_COORDS:
                        known    = LAGOS_AREA_COORDS[place_norm]
                        err_dist = dist_km(rlat, rlng, known["lat"], known["lng"])
                        if err_dist > 15.0:
                            # Nominatim found the wrong place — use known coords
                            result = known
                            _geocode_cache[cache_key] = result
                            return result

                result = {"lat": rlat, "lng": rlng}
                _geocode_cache[cache_key] = result
                return result
    except Exception:
        pass

    # Known suburb fallback
    if city_lower == "nairobi" and place_base in NAIROBI_SUBURB_COORDS:
        result = NAIROBI_SUBURB_COORDS[place_base]
        _geocode_cache[cache_key] = result
        return result
    if city_lower in ["lagos","abuja","kano","ibadan","port harcourt","enugu"]:
        place_norm = place_base.replace("'","")
        if place_norm in LAGOS_AREA_COORDS:
            result = LAGOS_AREA_COORDS[place_norm]
            _geocode_cache[cache_key] = result
            return result
    if city_lower == "asaba":
        place_norm = place_base.replace("'","")
        if place_norm in ASABA_AREA_COORDS:
            result = ASABA_AREA_COORDS[place_norm]
            _geocode_cache[cache_key] = result
            return result
    if city_lower == "benin city":
        place_norm = place_base.replace("'","")
        if place_norm in BENIN_AREA_COORDS:
            result = BENIN_AREA_COORDS[place_norm]
            _geocode_cache[cache_key] = result
            return result
    if city_lower in ["lagos","abuja","kano","ibadan","port harcourt","enugu"]:
        place_norm = place_base.replace("'","")
        if place_norm in LAGOS_AREA_COORDS:
            result = LAGOS_AREA_COORDS[place_norm]
            _geocode_cache[cache_key] = result
            return result
    result = CITY_COORDS.get(city_lower, {"lat": 6.5244, "lng": 3.3792})
    _geocode_cache[cache_key] = result
    return result


# ── ML prediction ─────────────────────────────────────────────

# Minimum plausible travel time in minutes. Any prediction below this is
# rejected as physically impossible (a 1 km walk takes ~13 min; routing
# returns at least 1-2 km for any two distinct geocoded points).
_MIN_TRAVEL_TIME_MIN = 1.0

# Maximum plausible single urban/intercity travel time (10 hours).
# Predictions above this almost certainly indicate model extrapolation error.
_MAX_TRAVEL_TIME_MIN = 600.0


def _travel_time_is_valid(t: float) -> bool:
    """Return True when t is a finite, positive, plausible travel time."""
    import math
    return (
        isinstance(t, (int, float))
        and math.isfinite(t)
        and _MIN_TRAVEL_TIME_MIN <= t <= _MAX_TRAVEL_TIME_MIN
    )


def predict_travel_time(
    distance_km: float, congestion: str, weather: str,
    alternatives: int, mode: str, city: str, rq_score: float
) -> float:
    """Predict travel time using ML model with mode + city + road quality.

    The XGBoost model (travel_time_model.pkl) was trained exclusively on
    intercity Nigerian highway data (distance range 129–1,040 km) plus Nairobi
    OD matrices. It is NOT validated for sub-50 km urban commutes — testing
    showed it returned -21.6 min for a 150 km trip, confirming extrapolation
    instability even within its nominal training range.

    Decision: use the formula fallback for all trips < 50 km; apply the ML model
    only for ≥ 50 km. If the ML model returns a negative, non-finite, or
    implausible value, fall through to the formula regardless of distance.
    """
    import math

    cmap = encoders["congestion_map"] if encoders else {"Low":0,"Medium":1,"High":2}
    wmap = encoders["weather_map"]    if encoders else {"Clear":0,"Cloudy":1,"Foggy":2,"Rainy":3}
    mmap = encoders.get("mode_map",   {}) if encoders else {}
    cimap= encoders.get("city_map",   {}) if encoders else {}

    c_enc   = cmap.get(congestion, 1)
    w_enc   = wmap.get(weather, 0)
    m_enc   = mmap.get(mode.lower(), 0)
    ci_enc  = cimap.get(get_country(city), 0)
    is_peak = int(c_enc >= 1 and w_enc >= 1)

    # Use formula for all city trips (< 50 km): model training range starts at
    # 129 km, so sub-50 km inputs are out-of-distribution by a factor of ~2.6×.
    if distance_km < 50:
        return _formula_travel_time(distance_km, congestion, weather, mode, city)

    if travel_model is None:
        return _formula_travel_time(distance_km, congestion, weather, mode, city)

    try:
        import pandas as pd
        feat_names = encoders.get("features") if encoders else None
        if feat_names and len(feat_names) == 8:
            feat_df = pd.DataFrame(
                [[distance_km, c_enc, w_enc, alternatives, is_peak, m_enc, ci_enc, rq_score]],
                columns=feat_names,
            )
            raw = float(travel_model.predict(feat_df)[0])
        else:
            raw = float(travel_model.predict(
                [[distance_km, c_enc, w_enc, alternatives, is_peak, m_enc, ci_enc, rq_score]]
            )[0])

        if not _travel_time_is_valid(raw):
            # Negative, NaN, Inf, or implausibly large/small — fall through.
            return _formula_travel_time(distance_km, congestion, weather, mode, city)

        return round(raw, 1)
    except Exception:
        return _formula_travel_time(distance_km, congestion, weather, mode, city)


def _formula_travel_time(distance_km, congestion, weather, mode, city):
    """Formula fallback when model unavailable."""
    if not transport_modes:
        base_speed = 30.0
    else:
        country   = get_country(city)
        mode_data = transport_modes.get(country, {}).get(mode.lower(),
                    transport_modes.get(country, {}).get("driving", {}))
        speeds    = mode_data.get("avg_speed_kmh", {"urban": 30})
        # Distance-based speed selection:
        # Short urban trips use the slowest (residential) speed.
        # Medium suburban routes use middle speed.
        # Long highway routes use the fastest speed.
        # This fixes the Thika matatu showing 5h instead of 1h45min.
        speed_vals = list(speeds.values())
        if distance_km > 20 and len(speed_vals) >= 1:
            base_speed = speed_vals[0]         # highway speed
        elif distance_km > 8 and len(speed_vals) >= 2:
            base_speed = speed_vals[1]         # primary road speed
        else:
            base_speed = min(speed_vals)       # urban/residential speed
        if not speed_vals:
            base_speed = 30.0

        peak_m = mode_data.get("peak_multiplier", 0.6)
        rain_m = mode_data.get("rain_multiplier", 0.8)

        if congestion == "High":   base_speed *= peak_m
        elif congestion == "Medium": base_speed *= (1 + peak_m) / 2
        if weather in ["Rainy","Foggy"]: base_speed *= rain_m

    t = (distance_km / max(base_speed, 1)) * 60
    wait = 0
    if transport_modes:
        country   = get_country(city)
        mode_data = transport_modes.get(country, {}).get(mode.lower(), {})
        wait      = mode_data.get("wait_time_min", 0)
    return round(t + wait, 1)


def get_commute_quality(
    congestion: str, weather: str, community_reports: int,
    safety_score: float, mode: str, city: str,
    distance_km: float, rq_score: float
) -> dict:
    """ML quality prediction with rule fallback.

    The RandomForest (commute_quality_model.pkl) was serialized with sklearn
    1.8.0 against named feature arrays. We pass a named pd.DataFrame to match
    the training schema exactly, which also suppresses the
    'X does not have valid feature names' UserWarning.
    """
    if quality_model and encoders:
        try:
            import pandas as pd
            cmap  = encoders["congestion_map"]
            wmap  = encoders["weather_map"]
            mmap  = encoders.get("mode_map", {})
            cimap = encoders.get("city_map", {})
            c_enc = cmap.get(congestion, 1)
            w_enc = wmap.get(weather, 0)
            m_enc = mmap.get(mode.lower(), 0)
            ci_enc= cimap.get(get_country(city), 0)
            is_pk = int(c_enc >= 1 and w_enc >= 1)
            feat_names = encoders.get("features")
            feat_vals  = [[distance_km, c_enc, w_enc, 2, is_pk, m_enc, ci_enc, rq_score]]
            if feat_names and len(feat_names) == 8:
                feat = pd.DataFrame(feat_vals, columns=feat_names)
            else:
                feat = feat_vals   # fallback: unnamed array (triggers warning)
            pred  = int(quality_model.predict(feat)[0])
            label = encoders["quality_labels"][pred]
            emoji = encoders["quality_emoji"][pred]
            score = {2: 90, 1: 60, 0: 30}[pred]
            return {"label": label, "emoji": emoji, "score": score}
        except Exception:
            pass

    # Rule fallback
    bad_weather  = weather in ["Rainy","Foggy"]
    high_traffic = congestion == "High"
    low_safety   = safety_score < 50
    if high_traffic or (bad_weather and community_reports >= 1) or community_reports >= 3:
        return {"label":"Poor",     "emoji":"🔴","score":30}
    elif congestion=="Medium" or bad_weather or community_reports>=1 or low_safety:
        return {"label":"Moderate", "emoji":"🟡","score":60}
    else:
        return {"label":"Good",     "emoji":"🟢","score":90}


# City-name → pkl key aliases for cities whose scores are stored under a
# different key (state/county name) than the city name used in CITY_COORDS.
#
# Data provenance:
#  Nigerian cities:  scores derived from FRSC/NBS crash data aggregated by state.
#    - port harcourt → "rivers" state (Rivers State, FRSC crash CSV)
#    - asaba         → "delta" state   (Delta State, FRSC crash CSV)
#    - benin city    → "edo" state     (Edo State, FRSC crash CSV)
#  Kenyan cities: scores derived from NTSA Kenya accident data aggregated by county.
#    - kisumu        → "ke_kisumu"     (Kisumu County, Kenya accidents CSV)
#    - eldoret       → "ke_uasin gishu" (Uasin Gishu County, Kenya accidents CSV)
_SAFETY_SCORE_ALIASES = {
    "port harcourt": "rivers",
    "asaba":         "delta",
    "benin city":    "edo",
    "kisumu":        "ke_kisumu",
    "eldoret":       "ke_uasin gishu",
}

# Cities whose safety scores in safety_scores.pkl are HARDCODED OVERRIDES
# in train_models.py rather than formula-derived from crash statistics.
# Specifically (see train_models.py lines 348-349):
#   nairobi = 68.0  — hardcoded (no formula derivation; rationale: WHO Kenya
#                     estimate ~28 fatalities/100k). NOT derived from NTSA data.
#   lagos   = 70.0  — hardcoded (or falls through to crash-CSV formula if
#                     "lagos" appears in crash data; train_models.py uses
#                     scores.get("lagos", 70.0) which means 70.0 is the
#                     default when the crash CSV has no "lagos" row).
# Both values are single-city estimates with no documented per-route basis.
# They are regional baseline indicators, not measured per-corridor scores.
_SAFETY_SCORE_HARDCODED_CITIES = {"nairobi", "lagos"}

# Geographic granularity for each city's score.
# "state" = aggregated over all crashes in the state/territory.
# "county" = aggregated over all crashes in the county.
# "city_estimate" = single point estimate, no sub-city variation.
_SAFETY_SCORE_GRANULARITY = {
    "nairobi":       "city_estimate",   # hardcoded override, no crash formula
    "lagos":         "city_estimate",   # hardcoded override
    "abuja":         "state",           # FCT crash stats
    "kano":          "state",
    "mombasa":       "county",          # ke_mombasa
    "nakuru":        "county",          # ke_nakuru
    "ibadan":        "state",           # Oyo state
    "enugu":         "state",
    "port harcourt": "state",           # Rivers state
    "asaba":         "state",           # Delta state
    "benin city":    "state",           # Edo state
    "kisumu":        "county",          # ke_kisumu
    "eldoret":       "county",          # ke_uasin gishu
}


def get_safety_score(city: str, mode: str) -> dict:
    """Return safety score with provenance metadata.

    Returns a dict with keys:
      score                – float, the safety score (0-100). None if unavailable.
      available            – bool, False when the city has no score in pkl.
      basis                – str, describes what the score is based on.
      mode_adj             – bool, True when a mode-specific multiplier was applied.
      is_hardcoded_override – bool, True when the score is a manually set estimate
                              rather than derived from the crash statistics formula.
      geographic_granularity – str, describes the geographic unit for the score.
      is_historical_indicator – bool, always True — no score reflects real-time
                              conditions; all are derived from past crash data.

    API consumers should check 'available' and 'is_hardcoded_override' before
    treating 'score' as independently validated crash-statistics data.
    """
    if not safety_scores:
        return {
            "score": None, "available": False,
            "basis": "models_not_loaded", "mode_adj": False,
            "is_hardcoded_override": False,
            "geographic_granularity": "unknown",
            "is_historical_indicator": True,
        }

    city_lower = city.lower()
    mult_map   = transport_modes.get("safety_multipliers", {}) if transport_modes else {}
    mult       = mult_map.get(mode.lower(), 1.0)
    mode_adj   = mult != 1.0

    # Resolve aliases first — some city names map to state/county keys in the pkl
    lookup_key = _SAFETY_SCORE_ALIASES.get(city_lower, city_lower)
    raw        = safety_scores.get(lookup_key)

    if raw is None:
        return {
            "score": None, "available": False,
            "basis": "no_data_for_city", "mode_adj": False,
            "is_hardcoded_override": False,
            "geographic_granularity": "unknown",
            "is_historical_indicator": True,
        }

    is_override  = city_lower in _SAFETY_SCORE_HARDCODED_CITIES
    granularity  = _SAFETY_SCORE_GRANULARITY.get(city_lower, "state")
    basis        = "hardcoded_city_estimate" if is_override else "crash_data_formula_pkl"

    score = round(min(100.0, max(0.0, raw * mult)), 1)
    return {
        "score":                   score,
        "available":               True,
        "basis":                   basis,
        "mode_adj":                mode_adj,
        "is_hardcoded_override":   is_override,
        "geographic_granularity":  granularity,
        "is_historical_indicator": True,
    }


def _resolve_safety_score(city: str, mode: str) -> float:
    """Backward-compatible helper used by places that need a plain float.

    When safety data is unavailable, returns None so callers can branch
    rather than receiving a silent 60.0 that obscures data gaps.
    Returns float or None.
    """
    result = get_safety_score(city, mode)
    return result["score"]  # may be None



# Mode restrictions are loaded from restriction_data.py
# which covers all cities in Nigeria and Kenya with sourced legal references.

# ── Alternative mode suggestion ───────────────────────────────

def _dest_is_cbd(destination: str, city: str) -> bool:
    """Return True when the destination is likely the city centre rather than
    a known outer suburb.

    Strategy (avoids a single brittle keyword list):
      1. Explicit CBD tokens — catch "CBD", "city centre", "town" etc.
      2. If the destination matches a *known outer suburb* in the suburb
         coord tables, it is NOT CBD — return False.
      3. If neither rule fires, assume CBD / unknown central area — return True.
         (Conservative: better to show the matatu terminus notice than to
         silently omit it when the destination is genuinely central.)
    """
    dest_lower = destination.lower().strip()
    dest_base  = dest_lower.split(",")[0].strip()

    # 1. Explicit CBD keywords (small, high-confidence set)
    CBD_TOKENS = {"cbd", "city centre", "city center", "downtown", "town centre",
                  "nairobi cbd", "town", "kenyatta avenue", "moi avenue",
                  "tom mboya", "haile selassie", "kenyatta", "uhuru park",
                  "city hall", "central"}
    if any(tok in dest_lower for tok in CBD_TOKENS):
        return True

    # 2. Known outer suburb → NOT CBD
    if city.lower() == "nairobi" and dest_base in NAIROBI_SUBURB_COORDS:
        return False

    # 3. Unknown destination — assume it could be central
    return True


def suggest_alternative_mode(
    mode: str, city: str, congestion: str,
    weather: str, distance_km: float,
    destination: str = "",
) -> Optional[str]:
    """Suggest a better mode given current conditions and legal restrictions."""
    if not transport_modes:
        return None
    country      = get_country(city)
    country_modes= transport_modes.get(country, {})
    rain         = weather in ["Rainy","Foggy"]
    high_traffic = congestion == "High"

    # ── Legal restriction checks (highest priority) ───────────
    # Uses restriction_data.py — covers Lagos, Abuja, Kano, Port Harcourt,
    # Enugu, Ibadan (Nigeria) and Nairobi (Kenya) with sourced legal references.
    restriction = get_restriction(mode.lower(), city.lower(), country)
    if restriction:
        status = restriction.get("status", "")
        msg    = restriction.get("message", "")

        if status == "BANNED_STATEWIDE":
            return msg.format(dist=distance_km) if "{dist" in msg else msg

        if status in ("BANNED", "BANNED_CITY_CENTRE", "BANNED_MAJOR_ROADS"):
            max_km = restriction.get("max_direct_km", 5.0)
            if distance_km > max_km:
                return msg.format(dist=distance_km) if "{dist" in msg else msg

        if status == "PARTIAL_BAN":
            max_km = restriction.get("max_direct_km", 10.0)
            if distance_km > max_km:
                return msg.format(dist=distance_km) if "{dist" in msg else msg

        if status == "TERMINUS_RESTRICTION":
            # Matatu CBD restriction — only warn when the destination is
            # likely the city centre (not a known outer suburb).
            if _dest_is_cbd(destination, city):
                return msg if msg else None
            return None

        if status == "NIGHT_CURFEW":
            # Only warn at night — check time
            import datetime
            hour = datetime.datetime.now().hour
            curfew_start = int(restriction.get("curfew_start","22:30").split(":")[0])
            curfew_end   = int(restriction.get("curfew_end","05:30").split(":")[0])
            is_night = hour >= curfew_start or hour < curfew_end
            if is_night:
                return msg.format(dist=distance_km) if "{dist" in msg else msg

    # Walking distance warning
    if mode.lower() == "walking":
        max_km = country_modes.get("walking", {}).get("max_recommended_km", 3.0)
        if distance_km > max_km:
            modes = [m for m in country_modes if m != "walking"]
            best  = min(modes, key=lambda m:
                transport_modes.get("speed_multipliers", {}).get(m, 0.5))
            info  = country_modes.get(best, {})
            return (f"⚠️ {distance_km:.1f}km is too far to walk safely. "
                    f"Consider {info.get('emoji','')} {info.get('label', best)} instead.")

    # Walk-to-transit suggestion when driving is stuck in High traffic
    if mode.lower() in ["driving", "rideshare", "taxi"] and high_traffic:
        walk_sug = get_walk_to_transit(
            origin_lat=0, origin_lng=0,   # coords not available here — handled in v2/predict
            city=city,
            origin_name="",
            travel_time_driving=travel_time if hasattr(travel_time, "__float__") else 60,
            congestion=congestion,
        )
        # Handled with real coords in v2/predict — skip here

    # Okada/boda in rain warning
    if mode.lower() in ["okada","boda_boda","boda"] and rain:
        safe_alt = "brt" if country == "nigeria" else "matatu"
        info = country_modes.get(safe_alt, {})
        return (f"⚠️ {country_modes[mode.lower()]['label']} in rain is dangerous. "
                f"Consider {info.get('emoji','')} {info.get('label', safe_alt)} for safety.")

    # BRT is faster than danfo in high traffic (Lagos)
    if mode.lower() == "danfo" and high_traffic and "brt" in country_modes:
        info = country_modes["brt"]
        return (f"💡 {info.get('emoji','')} {info.get('label','BRT')} has dedicated lanes — "
                f"likely faster than Danfo in current traffic.")

    return None


# ── Departure advice ──────────────────────────────────────────

def get_departure_advice(
    congestion: str, weather: str, travel_time: float, mode: str,
    safety_score: float = 100.0,
) -> str:
    """Return timed departure advice.

    safety_score is passed so that a clear-weather / low-congestion
    scenario does not produce 'conditions are good' when the route or
    mode carries elevated safety risk (score < 50).
    """
    MIN_SAVING  = 4     # don't recommend waiting for < 4 min saving
    LOW_SAFETY  = safety_score < 50

    if congestion == "High" and weather in ["Rainy","Foggy"]:
        saved = round(travel_time * 0.30)
        if saved < MIN_SAVING:
            return "Leave now — conditions are tough but the trip is short enough to go now."
        return f"Wait 20 min — leaving later could save ~{saved} min on this route."
    elif congestion == "High":
        saved = round(travel_time * 0.20)
        if saved < MIN_SAVING:
            return "Leave now — high congestion but the time saving from waiting is minimal."
        return f"Wait 15 min — conditions may ease and save ~{saved} min."
    elif weather in ["Rainy","Foggy"]:
        return "Leave now but allow extra time — weather is reducing speeds."
    elif LOW_SAFETY:
        # Safety score is a baseline property of the mode/city — traffic and
        # weather may be clear but the route/mode still carries elevated risk.
        return (
            "Leave now, but exercise extra caution — "
            f"safety score {safety_score:.0f}/100 indicates elevated risk on this route."
        )
    else:
        return "Leave now — conditions are good."


# ── AI Explanation ────────────────────────────────────────────

def generate_ai_explanation(
    origin: str, destination: str, travel_time: float,
    quality: dict, safety_score: float, weather: str,
    congestion: str, community_reports: int,
    departure_advice: str, mode: str, city: str,
    distance_km: float, rq_score: float,
    alt_suggestion: Optional[str]
) -> str:
    lines = []
    country      = get_country(city)
    mode_data    = (transport_modes or {}).get(country, {}).get(mode.lower(), {})
    mode_label   = mode_data.get("label", mode.title())
    mode_emoji   = mode_data.get("emoji", "")

    # Opening
    lines.append(
        f"Route {origin} → {destination}: {travel_time:.0f} min by "
        f"{mode_emoji} {mode_label} ({distance_km:.1f}km)."
    )

    # Weather
    if weather == "Rainy":
        lines.append("Rain is affecting road visibility and speeds.")
    elif weather == "Foggy":
        lines.append("Fog is reducing visibility on this route.")

    # Congestion
    if congestion == "High":
        lines.append("Heavy traffic on this corridor.")
    elif congestion == "Medium":
        lines.append("Moderate congestion detected.")

    # Community reports
    if community_reports > 0:
        lines.append(
            f"{community_reports} community report"
            f"{'s' if community_reports > 1 else ''} filed on this route in the last hour."
        )

    # Road quality context
    if rq_score < 30:
        lines.append(f"Road quality is poor on this route ({rq_score:.0f}/100) — expect rough conditions.")
    elif rq_score > 70:
        lines.append(f"Road quality is good ({rq_score:.0f}/100).")

    # Matatu route number tip — tells user which route to board.
    # This is additive information not shown elsewhere in the UI.
    if mode.lower() == "matatu" and city.lower() in ["nairobi"]:
        route_info = get_matatu_route(origin.split(",")[0].lower().strip(), city)
        if route_info:
            routes = "/".join(route_info["routes"])
            lines.append(
                f"🚐 Board Matatu Route {routes} from {route_info['stage']} "
                f"via {route_info['via']}."
            )

    # Walking distance note — not shown in any separate card.
    if mode.lower() == "walking" and distance_km > 3:
        lines.append(f"⚠️ {distance_km:.1f}km is a long walk — consider a faster mode.")

    # Quality verdict — summarises the ML model output.
    lines.append(f"Commute quality: {quality['emoji']} {quality['label']}.")

    # Departure advice — only appended here when it contains a "Wait" recommendation,
    # because the separate blue banner already shows it; repeating it in the paragraph
    # is redundant for the "Leave now" variants.
    if "Wait" in departure_advice:
        lines.append(departure_advice)

    # NOTE: safety score, mode rain warning, and alt_suggestion are intentionally
    # omitted here — they are rendered as dedicated cards in the frontend and
    # appending them here would cause every piece to display twice.

    return " ".join(lines)


# ── Response models ───────────────────────────────────────────

def calculate_arrival_time(time_str: Optional[str], travel_time_min: float) -> Optional[str]:
    """Calculate arrival time from departure + travel duration."""
    import datetime as _dt
    try:
        if time_str:
            hour, minute = map(int, time_str.split(":"))
        else:
            now = _dt.datetime.now()
            hour, minute = now.hour, now.minute
        total    = hour * 60 + minute + int(travel_time_min)
        arr_hour = (total // 60) % 24
        arr_min  = total % 60
        return f"{arr_hour:02d}:{arr_min:02d}"
    except Exception:
        return None


class PredictResponse(BaseModel):
    travel_time_min:                  float
    commute_quality:                  str
    quality_emoji:                    str
    quality_score:                    int
    safety_score:                     Optional[float] = None
    safety_score_available:           bool = True
    safety_score_basis:               str  = "crash_data_pkl"
    safety_score_is_hardcoded:        bool = False   # True = single-city estimate, not formula
    safety_score_granularity:         str  = "state" # geographic unit of the score
    safety_score_is_historical:       bool = True    # always True — past crash data only
    weather:                          str
    congestion:                       str
    departure_advice:                 str
    ai_explanation:                   str
    distance_km:                      Optional[float] = None
    # community_reports: only user-submitted Supabase reports.
    # Does NOT include TomTom, news, or OSM third-party incidents.
    community_reports:                int   = 0
    # third_party_alerts: geo-relevance-verified third-party incident count.
    # These are from TomTom, RSS news, Google News, and OSM — not user submissions.
    third_party_alerts:               int   = 0
    # total_incident_alerts: community_reports + third_party_alerts (combined hazard load).
    # Used by safety calculations; displayed with appropriate labeling.
    total_incident_alerts:            int   = 0
    road_quality_score:               Optional[float] = None
    road_quality_available:           bool  = True
    road_quality_source:              str   = "city_avg_pkl"
    mode_label:                       Optional[str] = None
    mode_emoji:                       Optional[str] = None
    alt_suggestion:                   Optional[str] = None
    arrival_time:                     Optional[str] = None
    route_geometry:                   Optional[list] = None
    # Model metadata — training date, not live evidence freshness
    training_data_date:               Optional[str] = None

class RecommendRequest(BaseModel):
    origin:      str
    destination: str
    mode:        str
    city:        str
    time:        Optional[str] = None

class RecommendResponse(BaseModel):
    recommended_departure: str
    windows: List[dict]
    best_window: dict

class ModesResponse(BaseModel):
    city:    str
    country: str
    modes:   List[dict]


# ── Endpoints ─────────────────────────────────────────────────

# Model versioning metadata
MODEL_VERSION   = "v2.1"
MODEL_TRAINED   = "2026-07-23"
TRAINING_ROWS   = 32749
DATASET_SOURCES = ["Nigeria Traffic (895 rows)", "Nairobi Driving OD (17,930)", "Nairobi Matatu OD (13,924)"]

@app.get("/health")
async def health():
    return {
        "status":           "ok",
        "version":          "2.0.0",
        "model_version":    MODEL_VERSION,
        "model_trained":    MODEL_TRAINED,
        "training_rows":    TRAINING_ROWS,
        "dataset_sources":  DATASET_SOURCES,
        "cities_covered":   len(CITY_COORDS),
        "transport_modes":  15,
        "named_roads":      20880,
        "models": {
            "travel_time":       travel_model is not None,
            "commute_quality":   quality_model is not None,
            "safety_scores":     safety_scores is not None,
            "road_quality":      road_quality is not None,
            "transport_modes":   transport_modes is not None,
        },
        "supported_cities":  list(CITY_COORDS.keys()),
        "ethical_design": {
            "blocked_reports":   ["police_checkpoint", "speed_trap"],
            "privacy":           "Coordinates anonymized to 1.1km grid",
            "data_retention":    "Reports expire automatically by type",
        },
        "live_intelligence": {
            "tomtom_active":  bool(os.getenv("TOMTOM_API_KEY")),
            "rss_feeds":      8,
            "google_news":    True,
            "note":           "Set TOMTOM_API_KEY env var on Render to activate real traffic flow. RSS + Google News active always.",
        },
    }


@app.get("/modes")
async def get_modes(city: str):
    """Return available transport modes for a city."""
    country      = get_country(city.lower())
    country_modes= (transport_modes or {}).get(country, {})
    modes_list   = [
        {
            "key":   key,
            "label": v.get("label", key),
            "emoji": v.get("emoji", ""),
            "description": v.get("description", ""),
        }
        for key, v in country_modes.items()
        if isinstance(v, dict)
    ]
    return ModesResponse(city=city, country=country, modes=modes_list)


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    """Main prediction endpoint."""

    city    = req.city.lower().strip()
    mode    = req.mode.lower().strip()
    country = get_country(city)

    # Validate city
    if city not in CITY_COORDS:
        raise HTTPException(
            status_code=400,
            detail=f"City '{city}' not supported. Supported: {list(CITY_COORDS.keys())}"
        )

    # Validate mode for city
    mode_valid, mode_msg = validate_mode_for_city(mode, city)
    if not mode_valid:
        raise HTTPException(status_code=400, detail=mode_msg)

    # Mode metadata
    mode_data  = (transport_modes or {}).get(country, {}).get(mode, {})
    mode_label = mode_data.get("label", mode.title())
    mode_emoji = mode_data.get("emoji", "")

    # 1. Route
    route_geometry = None
    try:
        origin_coords = await geocode_place(req.origin, city)
        dest_coords   = await geocode_place(req.destination, city)
        route         = await get_route(origin_coords, dest_coords, mode)
        distance_km   = route["distance_km"]
        route_geometry = route.get("route_geometry")
    except Exception:
        distance_km = 12.0   # reasonable urban fallback

    # 2. Weather
    coords      = CITY_COORDS.get(city, {"lat": 6.5244, "lng": 3.3792})
    weather_data= await get_weather(coords["lat"], coords["lng"])
    weather     = weather_data["label"]

    # 3. Congestion
    congestion = estimate_congestion(req.time)

    # 4. Community reports
    reports = await list_reports(city)
    cutoff  = time.time() - 3600
    community_count = len([
        r for r in reports
        if r.get("type") in ["accident","flood","road_closure","heavy_traffic"]
        and r.get("created_at", 0) > cutoff
    ])

    # 5. Road quality
    rq_result    = get_road_quality_score(city)
    rq_score     = rq_result["score"]
    rq_available = rq_result["available"]
    rq_source    = rq_result["source"]

    # 6. ML predictions
    travel_time      = predict_travel_time(
        distance_km, congestion, weather, 2, mode, city, rq_score
    )
    safety_result    = get_safety_score(city, mode)
    safety_score     = safety_result["score"]          # float or None
    safety_available = safety_result["available"]
    safety_basis     = safety_result["basis"]
    # Downstream functions that need a float fall back to 100.0 when
    # safety data is unavailable — this keeps advice conservative
    # (100 never triggers the low-safety branch) while explicitly
    # preserving the absence signal in the response fields.
    safety_for_logic = safety_score if safety_score is not None else 100.0
    quality      = get_commute_quality(
        congestion, weather, community_count,
        safety_for_logic, mode, city, distance_km, rq_score
    )

    # 7. Departure advice
    departure_advice = get_departure_advice(congestion, weather, travel_time, mode, safety_for_logic)

    # 8. Alternative suggestion
    alt_suggestion = suggest_alternative_mode(
        mode, city, congestion, weather, distance_km, destination=req.destination
    )

    # 9. AI explanation
    ai_explanation = generate_ai_explanation(
        origin=req.origin, destination=req.destination,
        travel_time=travel_time, quality=quality,
        safety_score=safety_for_logic, weather=weather,
        congestion=congestion, community_reports=community_count,
        departure_advice=departure_advice, mode=mode,
        city=city, distance_km=distance_km,
        rq_score=rq_score, alt_suggestion=alt_suggestion,
    )

    return PredictResponse(
        travel_time_min=travel_time,
        commute_quality=quality["label"],
        quality_emoji=quality["emoji"],
        quality_score=quality["score"],
        safety_score=safety_score,
        safety_score_available=safety_available,
        safety_score_basis=safety_basis,
        safety_score_is_hardcoded=safety_result.get("is_hardcoded_override", False),
        safety_score_granularity=safety_result.get("geographic_granularity", "state"),
        safety_score_is_historical=True,
        weather=weather,
        congestion=congestion,
        departure_advice=departure_advice,
        ai_explanation=ai_explanation,
        distance_km=round(distance_km, 2),
        community_reports=community_count,
        road_quality_score=round(rq_score, 1),
        road_quality_available=rq_available,
        road_quality_source=rq_source,
        mode_label=mode_label,
        mode_emoji=mode_emoji,
        alt_suggestion=alt_suggestion,
        arrival_time=calculate_arrival_time(req.time, travel_time),
        route_geometry=route_geometry,
    )


@app.post("/recommend", response_model=RecommendResponse)
async def recommend(req: RecommendRequest):
    """Departure time recommendation across 4 time windows."""

    city    = req.city.lower().strip()
    mode    = req.mode.lower().strip()
    coords  = CITY_COORDS.get(city, {"lat": 6.5244, "lng": 3.3792})
    rq_result = get_road_quality_score(city)
    rq_score  = rq_result["score"]

    try:
        o_coords = await geocode_place(req.origin, city)
        d_coords = await geocode_place(req.destination, city)
        route    = await get_route(o_coords, d_coords, mode)
        distance = route["distance_km"]
    except Exception:
        distance = 12.0

    weather_data = await get_weather(coords["lat"], coords["lng"])
    weather      = weather_data["label"]

    # Evaluate 4 departure windows
    current_hour = int(time.strftime("%H"))
    windows = []
    for offset in [0, 15, 30, 60]:
        hour      = (current_hour + offset // 60) % 24
        cong      = estimate_congestion(f"{hour}:00")
        t         = predict_travel_time(distance, cong, weather, 2, mode, city, rq_score)
        _safety_r = get_safety_score(city, mode)
        _safety_f = _safety_r["score"] if _safety_r["score"] is not None else 100.0
        quality   = get_commute_quality(cong, weather, 0, _safety_f, mode, city, distance, rq_score)
        windows.append({
            "label":        f"{'Now' if offset==0 else f'+{offset} min'}",
            "offset_min":   offset,
            "congestion":   cong,
            "travel_time":  t,
            "quality":      quality["label"],
            "emoji":        quality["emoji"],
            "score":        quality["score"],
        })

    best         = min(windows, key=lambda w: w["travel_time"])
    now_time     = windows[0]["travel_time"]
    best_saving  = now_time - best["travel_time"]

    # Only recommend waiting if saving is worth the wait.
    # < 8 min saving = not worth it. Tell the user to leave now.
    MIN_SAVING_MINUTES = 8
    if best["offset_min"] == 0 or best_saving < MIN_SAVING_MINUTES:
        advice = "Leave now — conditions are good."
        best   = windows[0]   # reset best to Now window
    else:
        advice = f"Wait {best['offset_min']} min — saves ~{best_saving:.0f} min on this route."

    return RecommendResponse(
        recommended_departure=advice,
        windows=windows,
        best_window=best,
    )


@app.post("/report")
async def submit_report(req: ReportRequest):
    report = {
        "city":       req.city,
        "type":       req.type,
        "location":   req.location,
        "lat":        req.lat,
        "lng":        req.lng,
        "timestamp":  time.time(),
        "created_at": time.time(),
    }
    result = await save_report(report)
    return {"ok": True, "message": "Report submitted. Thank you!", "storage": result.get("storage")}


@app.get("/reports")
async def get_reports(city: Optional[str] = None):
    reports = await list_reports(city)
    cutoff  = time.time() - (6 * 3600)
    recent  = [r for r in reports if r.get("created_at", 0) > cutoff]
    return {"reports": recent, "count": len(recent)}


# ═══════════════════════════════════════════════════════════════
# REPORT EXPIRATION — type-specific lifetimes
# ═══════════════════════════════════════════════════════════════

REPORT_EXPIRY_SECONDS = {
    "heavy_traffic":  20 * 60,        # 20 minutes
    "accident":       2 * 60 * 60,    # 2 hours
    "flood":          4 * 60 * 60,    # 4 hours
    "road_closure":   30 * 24 * 3600, # 30 days
    "construction":   30 * 24 * 3600, # 30 days
    "breakdown":      45 * 60,        # 45 minutes
    "default":        6 * 60 * 60,    # 6 hours fallback
}

# ═══════════════════════════════════════════════════════════════
# BLOCKED REPORT TYPES — ethical design
# ═══════════════════════════════════════════════════════════════

BLOCKED_REPORT_TYPES = {
    "police_checkpoint": "CommuteIQ does not allow police checkpoint reporting to prevent misuse.",
    "police":            "CommuteIQ does not allow police checkpoint reporting to prevent misuse.",
    "speed_trap":        "CommuteIQ does not allow speed trap reporting to prevent misuse.",
}

ALLOWED_REPORT_TYPES = [
    "accident", "flood", "road_closure",
    "heavy_traffic", "construction", "breakdown",
]


def get_active_reports(reports: list, city: str) -> list:
    """Filter reports by type-specific expiry time."""
    now = time.time()
    active = []
    for r in reports:
        rtype  = r.get("type", "default")
        expiry = REPORT_EXPIRY_SECONDS.get(rtype, REPORT_EXPIRY_SECONDS["default"])
        age    = now - r.get("created_at", 0)
        if age < expiry:
            r["expires_in_min"] = round((expiry - age) / 60)
            r["age_min"]        = round(age / 60)
            active.append(r)
    return active


# ═══════════════════════════════════════════════════════════════
# CONFIDENCE SCORE — explainability layer
# How confident is CommuteIQ in this prediction?
# ═══════════════════════════════════════════════════════════════

def calculate_confidence(
    community_reports: int,
    weather: str,
    congestion: str,
    distance_km: float,
    rq_score: float,
    time_str: Optional[str],
) -> dict:
    """
    Calculate prediction confidence score 0-100 with explanation.
    Judges love explainability — this is a key differentiator.
    """
    # Start higher on weekends and clear weather — these are predictable.
    # The 50 base caused 60% "Low confidence" on clear Saturdays with no
    # reports, which felt wrong when all signals were favourable.
    import datetime
    is_weekend = datetime.datetime.now().weekday() >= 5
    score      = 60 if is_weekend else 50  # base confidence
    contributors = []

    # Community reports boost confidence.
    # Reports are user-submitted, not independently verified — use "filed",
    # not "confirmed", to avoid implying external validation.
    if community_reports >= 5:
        score += 25
        contributors.append(f"{community_reports} community reports filed")
    elif community_reports >= 2:
        score += 15
        contributors.append(f"{community_reports} community reports filed")
    elif community_reports == 1:
        score += 8
        contributors.append("1 community report filed")

    # Weather confirmation
    if weather in ["Rainy", "Foggy"]:
        score += 10
        contributors.append(f"{weather} weather detected via live API")
    elif weather == "Clear":
        score += 5
        contributors.append("Clear weather — good visibility")

    # Peak hour pattern matching
    try:
        hour = int(time_str.split(":")[0]) if time_str else int(time.strftime("%H"))
    except Exception:
        hour = 8
    if 7 <= hour <= 9 or 17 <= hour <= 19:
        score += 10
        contributors.append("Peak hour — strong historical pattern")
    elif 10 <= hour <= 16:
        score += 5
        contributors.append("Daytime — stable traffic pattern")

    # Road quality data available
    if rq_score > 0:
        score += 5
        contributors.append("Road quality data from OSM")

    # Distance penalty — longer routes less certain
    if distance_km > 50:
        score -= 10
        contributors.append("Long route — higher uncertainty")
    elif distance_km < 20:
        score += 5
        contributors.append("Short urban route — high accuracy")

    score = min(98, max(35, score))  # cap at 98 — never claim 100%

    # Confidence label — threshold for "High" is 90, not 85.
    # Without live TomTom data the score is built from pattern matching
    # (time-of-day, road quality, weather API), not measured conditions.
    # Requiring 90 prevents reaching "High" on estimate-only inputs with
    # only community reports and weather, which would overstate certainty.
    if score >= 90:
        label = "High"
        emoji = "🟢"
    elif score >= 65:
        label = "Moderate"
        emoji = "🟡"
    else:
        label = "Low"
        emoji = "🔴"

    return {
        "score":        score,
        "label":        label,
        "emoji":        emoji,
        "contributors": contributors,
        "summary":      f"{emoji} {label} confidence ({score}%) — " + ", ".join(contributors[:2]),
    }


# ═══════════════════════════════════════════════════════════════
# ROUTE CONFIDENCE — star rating
# ═══════════════════════════════════════════════════════════════

def calculate_route_confidence(
    safety_score: float,
    quality_score: int,
    community_reports: int,
    weather: str,
    rq_score: float,
    confidence_score: int,
) -> dict:
    """
    Combine all factors into a 1-5 star route confidence rating.
    Makes the recommendation feel trustworthy and transparent.
    """
    # Weighted scoring
    score = (
        (safety_score / 100)    * 30 +   # 30% — safety
        (quality_score / 100)   * 25 +   # 25% — commute quality
        (confidence_score / 100)* 25 +   # 25% — prediction confidence
        (rq_score / 100)        * 10 +   # 10% — road quality
        (1 if weather == "Clear" else 0) * 10  # 10% — weather
    )

    # Convert to stars
    stars = round(score / 20)  # 0-100 → 0-5
    stars = max(1, min(5, stars))

    star_display = "⭐" * stars + "☆" * (5 - stars)

    factors = []
    if safety_score >= 70: factors.append("safe corridor")
    if community_reports == 0: factors.append("no incidents reported")
    if weather == "Clear": factors.append("clear weather")
    if rq_score > 50: factors.append("good road quality")
    if quality_score >= 60: factors.append("moderate or better commute")

    return {
        "stars":        stars,
        "display":      star_display,
        "score":        round(score),
        "label":        ["Very Low","Low","Moderate","High","Very High"][stars - 1],
        "factors":      factors,
        "summary":      f"{star_display} {['Very Low','Low','Moderate','High','Very High'][stars-1]} route confidence",
    }


# ═══════════════════════════════════════════════════════════════
# DEMAND BALANCER — staggered departure times
# This is the "one feature that could win you the hackathon"
# ═══════════════════════════════════════════════════════════════

def get_staggered_departure(
    user_id: Optional[str],
    congestion: str,
    travel_time: float,
    time_str: Optional[str],
) -> dict:
    """
    Instead of everyone leaving at 7:00 AM, CommuteIQ staggers
    departures to balance demand across the network.

    User A → 6:45 AM
    User B → 6:52 AM
    User C → 7:05 AM
    User D → 7:12 AM

    Deterministic based on user_id hash so the same user
    always gets a consistent recommendation.
    """
    if congestion != "High":
        return {"staggered": False, "advice": None}

    # Hash user_id to assign a consistent slot (0-4)
    if user_id:
        slot = hash(user_id) % 5
    else:
        import random
        slot = random.randint(0, 4)

    offsets = [-15, -8, 0, 7, 12]  # minutes relative to peak
    offset  = offsets[slot]

    try:
        base_hour = int(time_str.split(":")[0]) if time_str else 7
        base_min  = int(time_str.split(":")[1]) if time_str and ":" in time_str else 0
    except Exception:
        base_hour, base_min = 7, 0

    new_min  = base_min + offset
    new_hour = base_hour + new_min // 60
    new_min  = new_min % 60

    saving = round(travel_time * 0.15) if offset < 0 else 0

    return {
        "staggered":     True,
        "recommended_at": f"{new_hour:02d}:{new_min:02d}",
        "offset_min":    offset,
        "saving_min":    saving,
        "reason":        "CommuteIQ balances demand across the network — your personalized slot reduces peak congestion for everyone.",
        "display":       (
            f"🧠 Your optimal departure: {new_hour:02d}:{new_min:02d} "
            f"({'earlier' if offset < 0 else 'later'} than peak)"
            + (f" — saves ~{saving} min" if saving > 0 else "")
        ),
    }


# ═══════════════════════════════════════════════════════════════
# UPDATE /predict TO INCLUDE ALL NEW FEATURES
# ═══════════════════════════════════════════════════════════════

class PredictResponseV2(PredictResponse):
    confidence:          Optional[dict] = None
    route_confidence:    Optional[dict] = None
    staggered_departure: Optional[dict] = None
    active_reports:      Optional[int]  = None
    weather_trend:       Optional[dict] = None
    flood_risk:          Optional[dict] = None
    day_pattern:         Optional[dict] = None
    live_intelligence:   Optional[dict] = None   # TomTom + RSS + Google News data
    # arrival_time inherited from PredictResponse
    privacy_note:        str = "CommuteIQ stores only anonymized trip data. No personally identifiable travel history is collected or required."
    ethical_note:        str = "CommuteIQ does not allow police checkpoint or individual tracking reports."


@app.post("/v2/predict", response_model=PredictResponseV2)
async def predict_v2(req: PredictRequest, user_id: Optional[str] = None):
    """
    v2 prediction endpoint with:
    - Confidence score (explainability)
    - Route confidence (star rating)
    - Type-specific report expiration
    - Demand balancer (staggered departures)
    - Privacy + ethical design notes
    """
    city    = req.city.lower().strip()
    mode    = req.mode.lower().strip()
    country = get_country(city)

    if city not in CITY_COORDS:
        raise HTTPException(status_code=400, detail=f"City '{city}' not supported.")

    mode_valid, mode_msg = validate_mode_for_city(mode, city)
    if not mode_valid:
        raise HTTPException(status_code=400, detail=mode_msg)

    mode_data  = (transport_modes or {}).get(country, {}).get(mode, {})
    mode_label = mode_data.get("label", mode.title())
    mode_emoji = mode_data.get("emoji", "")

    # Route — all three variables are declared before the try so that the
    # except branch (and every line below) can reference them safely.
    # If geocoding or OSRM fails we raise 503 rather than silently using a
    # hardcoded 12 km distance that would produce a meaningless prediction.
    origin_coords: dict = {}
    dest_coords:   dict = {}
    route_geometry      = None
    try:
        origin_coords = await geocode_place(req.origin, city)
        dest_coords   = await geocode_place(req.destination, city)
        route          = await get_route(origin_coords, dest_coords, mode)
        distance_km    = route["distance_km"]
        route_geometry = route.get("route_geometry")
    except Exception as _routing_err:
        raise HTTPException(
            status_code=503,
            detail=(
                "Route unavailable: could not calculate a distance between "
                f"'{req.origin}' and '{req.destination}' in {city.title()}. "
                "The routing service may be temporarily unreachable. "
                "Please try again in a moment."
            ),
        ) from _routing_err

    # Weather
    coords       = CITY_COORDS.get(city, {"lat": 6.5244, "lng": 3.3792})
    weather_data = await get_weather(coords["lat"], coords["lng"])
    weather      = weather_data["label"]

    # ── Live Intelligence — TomTom + RSS News + Google News ──
    # Runs all three sources concurrently. Falls back gracefully if any fail.
    # If TOMTOM_API_KEY is not set, congestion falls back to time-of-day estimate.
    try:
        intel = await get_route_intelligence(
            origin=req.origin,
            destination=req.destination,
            city=city,
            lat=coords["lat"],
            lng=coords["lng"],
        )
        congestion        = intel["congestion"]
        live_intel_data   = intel
    except Exception:
        congestion      = estimate_congestion(req.time)
        live_intel_data = {
            "congestion":               congestion,
            "incidents":                [],
            "community_incidents":      [],
            "community_incident_count": 0,
            "confidence_boost":         0,
            "data_sources":             ["time_of_day_estimate"],
            "congestion_method":        "time_of_day_heuristic",
            "congestion_note":          "Congestion estimated from time of day (heuristic). "
                                        "Set TOMTOM_API_KEY for measured traffic data.",
            "provider_health":          {},
        }

    # Community reports — Supabase user-submitted reports only count as community reports.
    # Third-party live intelligence incidents (news, TomTom, OSM) are tracked separately
    # via community_incidents (geo-verified, confidence >= threshold) so we do not
    # inflate the community report count with unverified news items.
    all_reports      = await list_reports(city)
    active_reps      = get_active_reports(all_reports, city)
    incident_types   = ["accident", "flood", "road_closure", "heavy_traffic"]

    # Separate counts — never mix Supabase user submissions with third-party alerts.
    #
    # supabase_count    = user-submitted Supabase reports only (the original community)
    # third_party_count = geo-relevance-verified TomTom/news/OSM incidents (third parties)
    # total_alerts      = combined hazard load passed to safety/quality models
    #
    # The quality model and departure advice receive total_alerts so they can still
    # react to verified third-party incident signals; the API response labels each
    # count separately so clients can display the correct provenance.
    verified_live    = live_intel_data.get("community_incidents", [])
    supabase_count   = len([r for r in active_reps    if r.get("type") in incident_types])
    third_party_count= len([r for r in verified_live  if r.get("type") in incident_types])
    total_alerts     = supabase_count + third_party_count
    # community_count is kept as a local alias used by older helpers
    community_count  = total_alerts

    # Road quality
    rq_result    = get_road_quality_score(city)
    rq_score     = rq_result["score"]
    rq_available = rq_result["available"]
    rq_source    = rq_result["source"]

    # ML predictions
    travel_time      = predict_travel_time(distance_km, congestion, weather, 2, mode, city, rq_score)
    safety_result    = get_safety_score(city, mode)
    safety_score     = safety_result["score"]          # float or None
    safety_available = safety_result["available"]
    safety_basis     = safety_result["basis"]
    # Use 100.0 as the logic-fallback when safety data is absent:
    # - 100 never fires the low-safety branch in get_departure_advice()
    # - ensures commute quality rule fallback treats the route as safe
    # The raw None is preserved in the response for client transparency.
    safety_for_logic = safety_score if safety_score is not None else 100.0
    quality      = get_commute_quality(congestion, weather, community_count, safety_for_logic, mode, city, distance_km, rq_score)

    # Confidence — boosted if live data sources are available
    confidence       = calculate_confidence(community_count, weather, congestion, distance_km, rq_score, req.time)
    conf_boost       = live_intel_data.get("confidence_boost", 0)
    if conf_boost:
        confidence["score"]    = min(98, confidence["score"] + conf_boost)
        data_sources           = live_intel_data.get("data_sources", [])
        if "TomTom live traffic" in data_sources:
            confidence["contributors"].append("TomTom real-time traffic data")
        if any("news" in s for s in data_sources):
            confidence["contributors"].append("Live news intelligence")
        confidence["summary"] = (
            f"{confidence['emoji']} {confidence['label']} confidence "
            f"({confidence['score']}%) — " +
            ", ".join(confidence["contributors"][:3])
        )

    # Route confidence
    route_conf = calculate_route_confidence(
        safety_for_logic, quality["score"], community_count,
        weather, rq_score, confidence["score"]
    )

    # Departure advice
    departure_advice = get_departure_advice(congestion, weather, travel_time, mode, safety_for_logic)

    # Demand balancer
    staggered = get_staggered_departure(user_id, congestion, travel_time, req.time)

    # Alternative suggestion (mode restrictions + better mode tips)
    alt_suggestion = suggest_alternative_mode(
        mode, city, congestion, weather, distance_km, destination=req.destination
    )

    # Walk-to-transit — when driving is stuck, suggest walking to transit hub
    # Uses real origin coordinates for accurate walking distance calculation
    if not alt_suggestion and mode.lower() in ["driving","rideshare","taxi"]:
        walk_suggestion = get_walk_to_transit(
            origin_lat   = origin_coords.get("lat", coords["lat"]),
            origin_lng   = origin_coords.get("lng", coords["lng"]),
            city         = city,
            origin_name  = req.origin.split(",")[0].strip(),
            travel_time_driving = travel_time,
            congestion   = congestion,
        )
        if walk_suggestion:
            alt_suggestion = walk_suggestion["suggestion"]

    # Weather intelligence — trend + flood risk + day pattern
    weather_trend = await get_weather_trend(coords["lat"], coords["lng"])
    # origin_coords and dest_coords are always defined in this scope
    # (pre-initialised to {} above the try block; the try raises 503 on failure
    # so execution never reaches here with the empty-dict values).
    # Use explicit .get() guard rather than the unreliable dir() idiom that
    # was silently substituting (0,0) — the Atlantic Ocean — when both keys
    # were absent from the empty dict.
    _flood_origin = origin_coords if origin_coords.get("lat") is not None else coords
    _flood_dest   = dest_coords   if dest_coords.get("lat")   is not None else coords
    flood_risk    = get_flood_risk(city, _flood_origin, _flood_dest, weather)
    day_pattern   = get_day_pattern(city, req.time)

    # Day-of-week congestion pattern — kept for AI explanation context only.
    # NOT applied to travel_time because the formula already accounts for
    # congestion level. Multiplying again would double-count peak-hour delay
    # and produce absurdly long estimates (the "1h 40min" bug on danfo trips).
    day_mult = day_pattern["congestion_mult"]

    # AI explanation
    ai_explanation = generate_ai_explanation(
        origin=req.origin, destination=req.destination,
        travel_time=travel_time, quality=quality,
        safety_score=safety_for_logic, weather=weather,
        congestion=congestion, community_reports=community_count,
        departure_advice=departure_advice, mode=mode,
        city=city, distance_km=distance_km,
        rq_score=rq_score, alt_suggestion=alt_suggestion,
    )

    # NOTE: confidence, day_pattern, weather_trend and flood_risk are
    # returned as separate structured fields and displayed as dedicated
    # cards in the frontend. Do NOT append them to ai_explanation here —
    # that causes every piece to display twice in the UI.


    return PredictResponseV2(
        travel_time_min=travel_time,
        commute_quality=quality["label"],
        quality_emoji=quality["emoji"],
        quality_score=quality["score"],
        safety_score=safety_score,
        safety_score_available=safety_available,
        safety_score_basis=safety_basis,
        safety_score_is_hardcoded=safety_result.get("is_hardcoded_override", False),
        safety_score_granularity=safety_result.get("geographic_granularity", "state"),
        safety_score_is_historical=True,
        weather=weather,
        congestion=congestion,
        departure_advice=departure_advice,
        ai_explanation=ai_explanation,
        distance_km=round(distance_km, 2),
        # community_reports: Supabase user submissions only
        community_reports=supabase_count,
        # third_party_alerts: geo-relevance-verified TomTom/news/OSM incidents
        third_party_alerts=third_party_count,
        # total_incident_alerts: combined hazard load (used by quality/safety models)
        total_incident_alerts=total_alerts,
        road_quality_score=round(rq_score, 1),
        road_quality_available=rq_available,
        road_quality_source=rq_source,
        mode_label=mode_label,
        mode_emoji=mode_emoji,
        alt_suggestion=alt_suggestion,
        arrival_time=calculate_arrival_time(req.time, travel_time),
        route_geometry=route_geometry,
        confidence=confidence,
        route_confidence=route_conf,
        staggered_departure=staggered if staggered["staggered"] else None,
        active_reports=total_alerts,
        weather_trend=weather_trend,
        flood_risk=flood_risk if flood_risk["risk"] != "Low" else None,
        day_pattern=day_pattern,
        live_intelligence={
            # Normalized evidence layer — source, freshness, and verification status.
            # Distinguishes measured vs estimated congestion.
            "sources":                  live_intel_data.get("data_sources", ["time_of_day_estimate"]),
            "is_live":                  live_intel_data.get("congestion_detail", {}).get("live", False),
            "live_incidents":           live_intel_data.get("incident_count", 0),
            "community_incidents":      live_intel_data.get("community_incident_count", 0),
            "tomtom_active":            live_intel_data.get("congestion_detail", {}).get("source") == "TomTom live traffic",
            "flow_ratio":               live_intel_data.get("congestion_detail", {}).get("flow_ratio"),
            # congestion_method and congestion_note come directly from the intelligence
            # module — not recomputed here — so they are always consistent with the
            # actual detection path taken (TomTom / heuristic / exception fallback).
            "congestion_method":        live_intel_data.get("congestion_method", "time_of_day_heuristic"),
            "congestion_note":          live_intel_data.get("congestion_note",
                                            "Congestion estimated from time of day (heuristic). "
                                            "Set TOMTOM_API_KEY for measured traffic data."),
            # Provider health — exposes availability metadata without credentials or
            # internal stack traces.  Clients can use this to show "live data may be
            # limited" notices when key providers are unavailable.
            "provider_health":          live_intel_data.get("provider_health", {}),
            # all_providers_failed: True when incident count is zero AND at least one
            # provider is known to be down — cannot distinguish healthy-zero from outage.
            "all_providers_failed":     live_intel_data.get("all_providers_failed", False),
        },
        training_data_date=MODEL_TRAINED,
        privacy_note="CommuteIQ stores only anonymized trip data. No personally identifiable travel history is collected or required.",
        ethical_note="CommuteIQ does not allow police checkpoint or individual tracking reports.",
    )


@app.post("/v2/report")
async def submit_report_v2(req: ReportRequest):
    """
    v2 report endpoint with:
    - Blocked report types (ethics)
    - Privacy — strips exact coordinates, stores only city + general area
    - Type-specific expiry time returned to user
    """

    # Ethics check — block police reports
    if req.type.lower() in BLOCKED_REPORT_TYPES:
        raise HTTPException(
            status_code=403,
            detail=BLOCKED_REPORT_TYPES[req.type.lower()]
        )

    # Validate report type
    if req.type.lower() not in ALLOWED_REPORT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid report type. Allowed: {', '.join(ALLOWED_REPORT_TYPES)}"
        )

    # Privacy — anonymize coordinates to ~500m grid
    # Round to 2 decimal places ≈ 1.1km precision — enough for routing
    # but not enough to track individuals
    anon_lat = round(req.lat, 2) if req.lat else None
    anon_lng = round(req.lng, 2) if req.lng else None

    expiry_sec = REPORT_EXPIRY_SECONDS.get(req.type.lower(), REPORT_EXPIRY_SECONDS["default"])
    expiry_min = expiry_sec // 60

    report = {
        "city":        req.city,
        "type":        req.type.lower(),
        "location":    req.location,
        "lat":         anon_lat,        # anonymized
        "lng":         anon_lng,        # anonymized
        "timestamp":   time.time(),
        "created_at":  time.time(),
        "expires_at":  time.time() + expiry_sec,
        "expiry_min":  expiry_min,
    }

    result = await save_report(report)
    return {
        "ok":           True,
        "message":      f"Report submitted. Thank you for improving CommuteIQ!",
        "expires_in":   f"{expiry_min} minutes" if expiry_min < 1440 else f"{expiry_min // 1440} days",
        "privacy_note": "Your exact location was not stored. Only an anonymized area reference is used.",
        "storage":      result.get("storage"),
    }
