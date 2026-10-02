"""
data/features.py
────────────────
Transforms raw Ergast DataFrames into a flat feature matrix
suitable for training the PyTorch model.

Features produced per (driver, race) row:
  1.  qual_position           – qualifying / grid position
  2.  champ_points_entering   – driver championship points before this race
  3.  champ_position_entering – driver championship standing before this race
  4.  con_points_entering     – constructor points before this race
  5.  con_position_entering   – constructor standing before this race
  6.  recent_avg_finish       – avg finish position over last N races (form)
  7.  recent_dnf_rate         – fraction of last N races that were DNFs
  8.  circuit_avg_finish      – driver's historical avg finish at this circuit
  9.  circuit_appearances     – how many times driver has raced here before
  10. normalized_grid         – grid position / number of drivers (0–1)

Target:
  finish_position (integer 1–20) — we treat this as a regression task
  and then rank predictions to produce a final finishing order.
"""

import os
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import RECENT_FORM_WINDOW
from data.weather import build_weather_df

DNF_STATUSES = {
    "Accident", "Collision", "Engine", "Gearbox", "Hydraulics",
    "Mechanical", "Retired", "Suspension", "Transmission", "Electrical",
    "Brakes", "Overheating", "Spun off", "Power Unit", "ERS",
}


def is_dnf(status: str) -> bool:
    return any(s.lower() in status.lower() for s in DNF_STATUSES)


def build_feature_matrix(raw: dict, schedule_df=None) -> pd.DataFrame:
    """
    Parameters
    ----------
    raw         : dict returned by fetch_data.load_all_years()
                  keys: 'results', 'qualifying', 'driver_standings',
                        'constructor_standings'
    schedule_df : optional DataFrame with [year, round, circuit_id, date]
                  used to fetch real weather data; if None, neutral defaults
                  are used instead.

    Returns
    -------
    pd.DataFrame  with feature columns + 'finish_position' target
    """
    results = raw["results"].copy()
    qual    = raw["qualifying"].copy()
    drv_st  = raw["driver_standings"].copy()
    con_st  = raw["constructor_standings"].copy()

    # ── 1. Merge qualifying into results ─────────────────────────────────────
    df = results.merge(
        qual[["year", "round", "driver_id", "qual_position"]],
        on=["year", "round", "driver_id"],
        how="left",
    )

    # Fall back to grid position if qualifying data missing
    df["qual_position"] = df["qual_position"].fillna(df["grid"])

    # ── 2. Merge driver standings (standings AFTER round r → entering r+1) ──
    drv_entering = drv_st.copy()
    drv_entering["round"] = drv_entering["after_round"] + 1   # shift by 1
    drv_entering = drv_entering.rename(columns={
        "champ_points":   "champ_points_entering",
        "champ_position": "champ_position_entering",
    })[["year", "round", "driver_id", "champ_points_entering", "champ_position_entering"]]

    df = df.merge(drv_entering, on=["year", "round", "driver_id"], how="left")
    df["champ_points_entering"]   = df["champ_points_entering"].fillna(0)
    df["champ_position_entering"] = df["champ_position_entering"].fillna(20)

    # ── 3. Merge constructor standings ───────────────────────────────────────
    con_entering = con_st.copy()
    con_entering["round"] = con_entering["after_round"] + 1
    con_entering = con_entering.rename(columns={
        "con_points":   "con_points_entering",
        "con_position": "con_position_entering",
    })[["year", "round", "constructor_id", "con_points_entering", "con_position_entering"]]

    df = df.merge(con_entering, on=["year", "round", "constructor_id"], how="left")
    df["con_points_entering"]   = df["con_points_entering"].fillna(0)
    df["con_position_entering"] = df["con_position_entering"].fillna(10)

    # ── 4. Recent form features (rolling window across time) ─────────────────
    df = df.sort_values(["driver_id", "year", "round"]).reset_index(drop=True)
    df["dnf"] = df["status"].apply(lambda s: 1 if is_dnf(str(s)) else 0)

    # Safe finish for rolling — DNF gets last place (20)
    df["safe_finish"] = df.apply(
        lambda r: 20 if r["dnf"] else (r["finish_position"] if pd.notna(r["finish_position"]) else 20),
        axis=1,
    )

    df["recent_avg_finish"] = (
        df.groupby("driver_id")["safe_finish"]
          .transform(lambda x: x.shift(1).rolling(RECENT_FORM_WINDOW, min_periods=1).mean())
    )
    df["recent_dnf_rate"] = (
        df.groupby("driver_id")["dnf"]
          .transform(lambda x: x.shift(1).rolling(RECENT_FORM_WINDOW, min_periods=1).mean())
    )

    # ── 5. Circuit-specific history ───────────────────────────────────────────
    circuit_hist = (
        df.groupby(["driver_id", "circuit_id"])["safe_finish"]
          .expanding()
          .mean()
          .reset_index(level=[0, 1], drop=True)
    )
    df["circuit_avg_finish"]  = circuit_hist.shift(1).fillna(10)   # prior races only
    df["circuit_appearances"] = (
        df.groupby(["driver_id", "circuit_id"]).cumcount()         # 0-indexed prior races
    )

    # ── 6. Normalized grid ────────────────────────────────────────────────────
    drivers_per_race = df.groupby(["year", "round"])["driver_id"].transform("count")
    df["normalized_grid"] = df["qual_position"] / drivers_per_race

    # ── 7. Weather features ───────────────────────────────────────────────────
    weather_df = build_weather_df(results, schedule_df)
    df = df.merge(
        weather_df[["year", "round", "air_temp_c", "precipitation_mm",
                    "wind_speed_kmh", "is_wet_race"]],
        on=["year", "round"],
        how="left",
    )
    df["air_temp_c"]       = df["air_temp_c"].fillna(20.0)
    df["precipitation_mm"] = df["precipitation_mm"].fillna(0.0)
    df["wind_speed_kmh"]   = df["wind_speed_kmh"].fillna(10.0)
    df["is_wet_race"]      = df["is_wet_race"].fillna(0).astype(int)

    # ── 8. Drop rows with no finish position (not useful for training) ────────
    df = df.dropna(subset=["finish_position"])
    df["finish_position"] = df["finish_position"].astype(int)

    return df


FEATURE_COLS = [
    "qual_position",
    "champ_points_entering",
    "champ_position_entering",
    "con_points_entering",
    "con_position_entering",
    "recent_avg_finish",
    "recent_dnf_rate",
    "circuit_avg_finish",
    "circuit_appearances",
    "normalized_grid",
    # weather features
    "air_temp_c",
    "precipitation_mm",
    "wind_speed_kmh",
    "is_wet_race",
]

TARGET_COL = "finish_position"


def get_X_y(df: pd.DataFrame):
    """Return numpy arrays X (features) and y (target) from a feature DataFrame."""
    X = df[FEATURE_COLS].fillna(df[FEATURE_COLS].median()).values.astype(np.float32)
    y = df[TARGET_COL].values.astype(np.float32)
    return X, y


def build_prediction_row(
    driver_id: str,
    constructor_id: str,
    qual_position: int,
    n_drivers: int,
    champ_points: float,
    champ_pos: int,
    con_points: float,
    con_pos: int,
    recent_avg_finish: float,
    recent_dnf_rate: float,
    circuit_avg_finish: float,
    circuit_appearances: int,
    air_temp_c: float = 20.0,
    precipitation_mm: float = 0.0,
    wind_speed_kmh: float = 10.0,
    is_wet_race: int = 0,
) -> np.ndarray:
    """
    Build a single feature vector for one driver in an upcoming race.
    Used at prediction time when there are no 'results' rows yet.
    """
    return np.array([
        qual_position,
        champ_points,
        champ_pos,
        con_points,
        con_pos,
        recent_avg_finish,
        recent_dnf_rate,
        circuit_avg_finish,
        circuit_appearances,
        qual_position / n_drivers,
        air_temp_c,
        precipitation_mm,
        wind_speed_kmh,
        is_wet_race,
    ], dtype=np.float32)
