"""
data/weather.py
───────────────
Fetches historical weather conditions for F1 race locations using the
Open-Meteo archive API (https://open-meteo.com/) — free, no key required.

Weather is sampled at the typical race-start hour (14:00 local time).
Features returned:
  air_temp_c      – air temperature at race start (°C)
  precipitation_mm – precipitation in the race-start hour (mm)
  wind_speed_kmh  – wind speed at race start (km/h)
  is_wet_race     – 1 if precipitation_mm > 0.5, else 0
"""

import os
import json
import time
import urllib.request
import urllib.parse
import urllib.error
import pandas as pd

# ── Circuit coordinates (lat, lon) ────────────────────────────────────────────
CIRCUIT_COORDS: dict[str, tuple[float, float]] = {
    "bahrain":          (26.0325,  50.5106),
    "bahrain_outer":    (26.0325,  50.5106),
    "jeddah":           (21.6319,  39.1044),
    "albert_park":      (-37.8497, 144.9680),
    "baku":             (40.3725,  49.8533),
    "miami":            (25.9581,  -80.2389),
    "monaco":           (43.7347,   7.4205),
    "catalunya":        (41.5700,   2.2611),
    "silverstone":      (52.0786,  -1.0169),
    "hungaroring":      (47.5830,  19.2526),
    "spa":              (50.4372,   5.9714),
    "zandvoort":        (52.3888,   4.5409),
    "monza":            (45.6156,   9.2811),
    "marina_bay":       ( 1.2914, 103.8639),
    "suzuka":           (34.8431, 136.5408),
    "losail":           (25.4900,  51.4542),
    "americas":         (30.1328,  -97.6411),
    "rodriguez":        (19.4042,  -99.0907),
    "interlagos":       (-23.7036, -46.6997),
    "vegas":            (36.1162, -115.1745),
    "yas_marina":       (24.4672,  54.6031),
    "nurburgring":      (50.3356,   6.9475),
    "hockenheimring":   (49.3278,   8.5656),
    "istanbul":         (40.9517,  29.4050),
    "sepang":           ( 2.7606, 101.7381),
    "shanghai":         (31.3389, 121.2200),
    "sochi":            (43.4057,  39.9717),
    "red_bull_ring":    (47.2197,  14.7647),
    "portimao":         (37.2277,  -8.6270),
    "mugello":          (43.9975,  11.3711),
    "imola":            (44.3439,  11.7167),
    "ricard":           (43.2506,   5.7917),
    "villeneuve":       (45.5000,  -73.5228),
    "kyalami":          (-25.9997,  28.0597),
    "yeongam":          (34.7333, 126.4167),
    "buddh":            (28.3487,  77.5331),
    "austin":           (30.1328,  -97.6411),
}

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
RACE_HOUR   = 14          # assume 14:00 local as representative race-start slot
_CACHE_FILE = os.path.join(os.path.dirname(__file__), "cache", "weather_cache.json")


def _load_cache() -> dict:
    if os.path.exists(_CACHE_FILE):
        with open(_CACHE_FILE, "r") as f:
            return json.load(f)
    return {}


def _save_cache(cache: dict) -> None:
    os.makedirs(os.path.dirname(_CACHE_FILE), exist_ok=True)
    with open(_CACHE_FILE, "w") as f:
        json.dump(cache, f)


def fetch_race_weather(circuit_id: str, date: str) -> dict:
    """
    Return weather dict for a specific circuit on a specific date.

    Parameters
    ----------
    circuit_id : Ergast circuit identifier (e.g. 'silverstone')
    date       : ISO date string 'YYYY-MM-DD'

    Returns
    -------
    dict with keys: air_temp_c, precipitation_mm, wind_speed_kmh, is_wet_race
    Falls back to neutral defaults if the circuit or date is unknown.
    """
    _default = {"air_temp_c": 20.0, "precipitation_mm": 0.0,
                "wind_speed_kmh": 10.0, "is_wet_race": 0}

    coords = CIRCUIT_COORDS.get(circuit_id)
    if coords is None or not date or date == "TBD":
        return _default

    cache_key = f"{circuit_id}_{date}"
    cache = _load_cache()
    if cache_key in cache:
        return cache[cache_key]

    lat, lon = coords
    params = urllib.parse.urlencode({
        "latitude":   lat,
        "longitude":  lon,
        "start_date": date,
        "end_date":   date,
        "hourly":     "temperature_2m,precipitation,windspeed_10m",
        "timezone":   "auto",
    })
    url = f"{ARCHIVE_URL}?{params}"

    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            data = json.loads(r.read().decode())

        hours = data["hourly"]["time"]
        temps = data["hourly"]["temperature_2m"]
        precip = data["hourly"]["precipitation"]
        wind  = data["hourly"]["windspeed_10m"]

        # find the hour closest to RACE_HOUR
        best_idx = 0
        for i, t in enumerate(hours):
            if int(t[11:13]) == RACE_HOUR:
                best_idx = i
                break

        result = {
            "air_temp_c":       round(float(temps[best_idx]  or 20.0), 1),
            "precipitation_mm": round(float(precip[best_idx] or 0.0),  2),
            "wind_speed_kmh":   round(float(wind[best_idx]   or 10.0), 1),
            "is_wet_race":      1 if (precip[best_idx] or 0) > 0.5 else 0,
        }
        cache[cache_key] = result
        _save_cache(cache)
        time.sleep(0.2)   # be polite to the free API
        return result

    except Exception:
        return _default


def build_weather_df(results_df: pd.DataFrame, schedule_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Build a DataFrame indexed by (year, round, circuit_id) with weather columns.
    `schedule_df` should have columns [year, round, circuit_id, date].

    If schedule_df is None, tries to infer from results_df (no date available,
    so neutral defaults are returned).
    """
    if schedule_df is None:
        # No dates available; return neutral defaults for every race
        races = results_df[["year", "round", "circuit_id"]].drop_duplicates()
        for col, val in [("air_temp_c", 20.0), ("precipitation_mm", 0.0),
                         ("wind_speed_kmh", 10.0), ("is_wet_race", 0)]:
            races[col] = val
        return races.reset_index(drop=True)

    rows = []
    for _, r in schedule_df.drop_duplicates(["year", "round"]).iterrows():
        w = fetch_race_weather(r["circuit_id"], r.get("date", "TBD"))
        rows.append({
            "year":             int(r["year"]),
            "round":            int(r["round"]),
            "circuit_id":       r["circuit_id"],
            **w,
        })
    return pd.DataFrame(rows)
