"""
data/fetch_data.py
──────────────────
Fetches historical F1 data from the Ergast API (jolpi.ca mirror) and
caches it locally as CSV files so we don't re-download on every run.

Endpoints used:
  • /f1/{year}/results.json      → race finishing positions
  • /f1/{year}/qualifying.json   → grid / qualifying positions
  • /f1/{year}/driverStandings.json  → championship points before each race
  • /f1/{year}/constructorStandings.json
"""

import os
import ssl
import time
import json
import urllib.request
import urllib.error
import pandas as pd
from tqdm import tqdm
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import ERGAST_BASE_URL, DATA_DIR

os.makedirs(DATA_DIR, exist_ok=True)

try:
    import certifi
    _SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    _SSL_CONTEXT = ssl.create_default_context()

# ── Low-level helper ──────────────────────────────────────────────────────────

def _get(url: str, retries: int = 3) -> dict:
    """GET a JSON endpoint with simple retry logic."""
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=15, context=_SSL_CONTEXT) as response:
                if response.getcode() != 200:
                    raise urllib.error.HTTPError(url, response.getcode(), response.reason, response.headers, None)
                data = response.read().decode("utf-8")
                return json.loads(data)
        except (urllib.error.HTTPError, urllib.error.URLError, ValueError) as e:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)


# ── Per-year fetchers ─────────────────────────────────────────────────────────

def _get_all_pages(base_url: str, page_size: int = 100) -> list:
    """Fetch all pages of a paginated Ergast endpoint and return the merged Races list."""
    all_races = []
    offset = 0
    while True:
        url  = f"{base_url}?limit={page_size}&offset={offset}"
        data = _get(url)
        mr   = data["MRData"]
        races = mr["RaceTable"]["Races"]
        all_races.extend(races)
        total = int(mr.get("total", 0))
        offset += page_size
        if offset >= total:
            break
    return all_races


def fetch_race_results(year: int) -> pd.DataFrame:
    """Return a DataFrame of every driver's result for every race in `year`."""
    cache = os.path.join(DATA_DIR, f"results_{year}.csv")
    if os.path.exists(cache):
        return pd.read_csv(cache)

    rows  = []
    races = _get_all_pages(f"{ERGAST_BASE_URL}/{year}/results.json")

    for race in races:
        circuit  = race["Circuit"]["circuitId"]
        round_no = int(race["round"])
        race_name = race["raceName"]
        for res in race["Results"]:
            driver = res["Driver"]["driverId"]
            constructor = res["Constructor"]["constructorId"]
            grid     = int(res.get("grid", 0))
            position = int(res["position"]) if res.get("position") else None
            points   = float(res.get("points", 0))
            status   = res.get("status", "")
            rows.append({
                "year": year,
                "round": round_no,
                "race_name": race_name,
                "circuit_id": circuit,
                "driver_id": driver,
                "constructor_id": constructor,
                "grid": grid,
                "finish_position": position,
                "points": points,
                "status": status,
            })

    df = pd.DataFrame(rows)
    df.to_csv(cache, index=False)
    return df


def fetch_qualifying(year: int) -> pd.DataFrame:
    """Return qualifying results (position + times) for every race in `year`."""
    cache = os.path.join(DATA_DIR, f"qualifying_{year}.csv")
    if os.path.exists(cache):
        return pd.read_csv(cache)

    rows  = []
    races = _get_all_pages(f"{ERGAST_BASE_URL}/{year}/qualifying.json")

    for race in races:
        round_no = int(race["round"])
        for q in race.get("QualifyingResults", []):
            rows.append({
                "year":           year,
                "round":          round_no,
                "driver_id":      q["Driver"]["driverId"],
                "constructor_id": q["Constructor"]["constructorId"],
                "qual_position":  int(q.get("position", 0)),
            })

    df = pd.DataFrame(rows)
    df.to_csv(cache, index=False)
    return df


def fetch_driver_standings_by_round(year: int, results_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Compute driver championship standings after each round by accumulating
    points from the results DataFrame. This avoids making 25 extra API
    requests per year (which caused rate-limiting errors).

    Falls back to the Ergast standings endpoint only if results_df is not provided.
    """
    cache = os.path.join(DATA_DIR, f"driver_standings_{year}.csv")
    if os.path.exists(cache):
        return pd.read_csv(cache)

    if results_df is not None:
        # Compute from results — much faster, no extra API calls
        yr = results_df[results_df["year"] == year].copy()
        yr = yr.sort_values(["round"])

        rows = []
        for round_no in sorted(yr["round"].unique()):
            up_to = yr[yr["round"] <= round_no]
            cum_pts = up_to.groupby("driver_id")["points"].sum().reset_index()
            cum_pts = cum_pts.sort_values("points", ascending=False).reset_index(drop=True)
            cum_pts["champ_position"] = cum_pts.index + 1
            for _, row in cum_pts.iterrows():
                rows.append({
                    "year":            year,
                    "after_round":     round_no,
                    "driver_id":       row["driver_id"],
                    "champ_points":    row["points"],
                    "champ_position":  row["champ_position"],
                })

        df = pd.DataFrame(rows)
        df.to_csv(cache, index=False)
        return df

    # Fallback: fetch from API with a delay between requests
    rows = []
    for round_no in range(1, 26):
        try:
            url  = f"{ERGAST_BASE_URL}/{year}/{round_no}/driverStandings.json"
            data = _get(url)
            standings_list = data["MRData"]["StandingsTable"].get("StandingsLists", [])
            if not standings_list:
                break
            for entry in standings_list[0]["DriverStandings"]:
                rows.append({
                    "year":           year,
                    "after_round":    round_no,
                    "driver_id":      entry["Driver"]["driverId"],
                    "champ_points":   float(entry["points"]),
                    "champ_position": int(entry["position"]),
                })
            time.sleep(0.3)   # be polite to the API
        except Exception:
            break

    df = pd.DataFrame(rows)
    df.to_csv(cache, index=False)
    return df


def fetch_constructor_standings_by_round(year: int, results_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Compute constructor championship standings after each round by
    accumulating points from results. Same approach as driver standings.
    """
    cache = os.path.join(DATA_DIR, f"constructor_standings_{year}.csv")
    if os.path.exists(cache):
        return pd.read_csv(cache)

    if results_df is not None:
        yr = results_df[results_df["year"] == year].copy()
        yr = yr.sort_values(["round"])

        rows = []
        for round_no in sorted(yr["round"].unique()):
            up_to = yr[yr["round"] <= round_no]
            cum_pts = up_to.groupby("constructor_id")["points"].sum().reset_index()
            cum_pts = cum_pts.sort_values("points", ascending=False).reset_index(drop=True)
            cum_pts["con_position"] = cum_pts.index + 1
            for _, row in cum_pts.iterrows():
                rows.append({
                    "year":           year,
                    "after_round":    round_no,
                    "constructor_id": row["constructor_id"],
                    "con_points":     row["points"],
                    "con_position":   row["con_position"],
                })

        df = pd.DataFrame(rows)
        df.to_csv(cache, index=False)
        return df

    # Fallback: fetch from API
    rows = []
    for round_no in range(1, 26):
        try:
            url  = f"{ERGAST_BASE_URL}/{year}/{round_no}/constructorStandings.json"
            data = _get(url)
            standings_list = data["MRData"]["StandingsTable"].get("StandingsLists", [])
            if not standings_list:
                break
            for entry in standings_list[0]["ConstructorStandings"]:
                rows.append({
                    "year":           year,
                    "after_round":    round_no,
                    "constructor_id": entry["Constructor"]["constructorId"],
                    "con_points":     float(entry["points"]),
                    "con_position":   int(entry["position"]),
                })
            time.sleep(0.3)
        except Exception:
            break

    df = pd.DataFrame(rows)
    df.to_csv(cache, index=False)
    return df


# ── Multi-year loader ─────────────────────────────────────────────────────────

def load_all_years(years: list[int]) -> dict[str, pd.DataFrame]:
    """
    Download (or load from cache) data for a list of years.
    Standings are computed from race results to avoid API rate limiting.
    """
    all_results    = []
    all_qualifying = []
    all_drv_stand  = []
    all_con_stand  = []

    for year in tqdm(years, desc="Fetching F1 data"):
        try:
            results = fetch_race_results(year)
            qual    = fetch_qualifying(year)
            drv_st  = fetch_driver_standings_by_round(year, results_df=results)
            con_st  = fetch_constructor_standings_by_round(year, results_df=results)
            all_results.append(results)
            all_qualifying.append(qual)
            all_drv_stand.append(drv_st)
            all_con_stand.append(con_st)
        except Exception as e:
            print(f"  WARNING Skipping year {year}: {type(e).__name__}: {e}")

    if not all_results:
        raise RuntimeError(
            "No data fetched. Check internet connection and that api.jolpi.ca is reachable."
        )

    return {
        "results":               pd.concat(all_results,    ignore_index=True),
        "qualifying":            pd.concat(all_qualifying, ignore_index=True),
        "driver_standings":      pd.concat(all_drv_stand,  ignore_index=True),
        "constructor_standings": pd.concat(all_con_stand,  ignore_index=True),
    }


def fetch_upcoming_race_info(year: int, round_no: int) -> dict:
    """Fetch metadata about a specific upcoming (or past) race."""
    url  = f"{ERGAST_BASE_URL}/{year}/{round_no}.json"
    data = _get(url)
    races = data["MRData"]["RaceTable"]["Races"]
    if not races:
        return {}
    race = races[0]
    return {
        "race_name":  race["raceName"],
        "circuit_id": race["Circuit"]["circuitId"],
        "circuit_name": race["Circuit"]["circuitName"],
        "country":    race["Circuit"]["Location"]["country"],
        "date":       race.get("date", "TBD"),
    }


def get_race_schedule(year: int) -> list[dict]:
    """Return the full race schedule for a given year."""
    url  = f"{ERGAST_BASE_URL}/{year}.json"
    data = _get(url)
    races = data["MRData"]["RaceTable"]["Races"]
    return [
        {
            "round":      int(r["round"]),
            "race_name":  r["raceName"],
            "circuit_id": r["Circuit"]["circuitId"],
            "country":    r["Circuit"]["Location"]["country"],
            "date":       r.get("date", "TBD"),
        }
        for r in races
    ]


def get_schedule_df(years: list[int]) -> pd.DataFrame:
    """Return a DataFrame of all races across multiple years, including dates."""
    cache = os.path.join(DATA_DIR, "schedule_all.csv")
    if os.path.exists(cache):
        df = pd.read_csv(cache)
        # Re-fetch if any requested year is missing
        cached_years = set(df["year"].unique())
        if all(y in cached_years for y in years):
            return df[df["year"].isin(years)].reset_index(drop=True)

    rows = []
    for year in years:
        for r in get_race_schedule(year):
            rows.append({"year": year, **r})
    df = pd.DataFrame(rows)
    # Merge/update cache
    if os.path.exists(cache):
        old = pd.read_csv(cache)
        df = pd.concat([old[~old["year"].isin(years)], df], ignore_index=True)
    df.to_csv(cache, index=False)
    return df[df["year"].isin(years)].reset_index(drop=True)
