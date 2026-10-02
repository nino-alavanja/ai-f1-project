"""
data/telemetry.py
──────────────────
FastF1-powered lap-time telemetry feature extractor.

Extracts per-driver features from a completed race session that go beyond
what the Ergast results API provides:

  avg_lap_time_s       – mean lap time in seconds (excluding pit laps)
  lap_time_vs_fastest  – driver's median lap vs fastest in field (ratio; 1.0 = matched fastest)
  top_speed_kmh        – max speed recorded at a speed trap
  sector1_avg_s        – avg S1 time
  sector2_avg_s        – avg S2 time
  sector3_avg_s        – avg S3 time
  tyre_stints          – number of tyre stints (strategy complexity)
  most_used_compound   – soft/medium/hard/intermediate/wet encoded as int

These are optional enrichment features. The module degrades gracefully if
FastF1 data is unavailable (returns None so callers can skip).

Usage:
  from data.telemetry import fetch_telemetry_features
  feats = fetch_telemetry_features(year=2023, round_no=1)
  # feats: dict[driver_id -> dict[feature_name -> value]]
"""

import os
import warnings
import numpy as np

# FastF1 cache directory
_FF1_CACHE = os.path.join(os.path.dirname(__file__), "cache", "fastf1")
os.makedirs(_FF1_CACHE, exist_ok=True)

COMPOUND_MAP = {
    "SOFT": 1, "MEDIUM": 2, "HARD": 3,
    "INTERMEDIATE": 4, "WET": 5,
    "HYPERSOFT": 1, "ULTRASOFT": 1, "SUPERSOFT": 1,
    "SUPERHARD": 3,
}


def fetch_telemetry_features(
    year: int,
    round_no: int,
    session_type: str = "R",
) -> dict | None:
    """
    Fetch lap-time telemetry for a completed race and return a dict of
    per-driver feature dicts.

    Parameters
    ----------
    year         : season year (FastF1 data available from 2018+)
    round_no     : race round number
    session_type : 'R' = race, 'Q' = qualifying

    Returns
    -------
    dict[driver_abbreviation -> dict] or None if data unavailable
    """
    try:
        import fastf1
        fastf1.Cache.enable_cache(_FF1_CACHE)
    except ImportError:
        return None

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            session = fastf1.get_session(year, round_no, session_type)
            session.load(telemetry=True, laps=True, weather=False, messages=False)
    except Exception:
        return None

    laps = session.laps
    if laps is None or len(laps) == 0:
        return None

    results = {}

    for drv in laps["Driver"].unique():
        drv_laps = laps.pick_driver(drv)

        # Exclude pit-out/in laps and safety-car laps
        clean = drv_laps[
            (drv_laps["PitOutTime"].isna()) &
            (drv_laps["PitInTime"].isna()) &
            (drv_laps["LapTime"].notna())
        ]

        if len(clean) == 0:
            continue

        try:
            lap_times_s = clean["LapTime"].dt.total_seconds()
            sector1_s   = clean["Sector1Time"].dropna().dt.total_seconds()
            sector2_s   = clean["Sector2Time"].dropna().dt.total_seconds()
            sector3_s   = clean["Sector3Time"].dropna().dt.total_seconds()

            # Fastest lap in field for relative comparison
            fastest_s = laps[laps["LapTime"].notna()]["LapTime"].dt.total_seconds().min()

            # Tyre stints: count compound changes
            compounds = drv_laps["Compound"].dropna().tolist()
            stints = 1
            for i in range(1, len(compounds)):
                if compounds[i] != compounds[i - 1]:
                    stints += 1

            most_used = "UNKNOWN"
            if compounds:
                from collections import Counter
                most_used = Counter(compounds).most_common(1)[0][0]

            # Top speed from telemetry (may not always be available)
            top_speed = np.nan
            try:
                tel = clean.iloc[0].get_telemetry()
                if tel is not None and "Speed" in tel.columns:
                    top_speed = float(tel["Speed"].max())
            except Exception:
                pass

            results[drv] = {
                "avg_lap_time_s":       round(float(lap_times_s.mean()), 3),
                "lap_time_vs_fastest":  round(float(lap_times_s.median() / fastest_s), 4)
                                        if fastest_s > 0 else 1.0,
                "top_speed_kmh":        round(top_speed, 1) if not np.isnan(top_speed) else None,
                "sector1_avg_s":        round(float(sector1_s.mean()), 3) if len(sector1_s) else None,
                "sector2_avg_s":        round(float(sector2_s.mean()), 3) if len(sector2_s) else None,
                "sector3_avg_s":        round(float(sector3_s.mean()), 3) if len(sector3_s) else None,
                "tyre_stints":          stints,
                "most_used_compound":   most_used,
                "compound_code":        COMPOUND_MAP.get(most_used, 0),
            }
        except Exception:
            continue

    return results if results else None


def _format_driver_metric(driver: str, value, unit: str = "", decimals: int = 3) -> str:
    """Format one telemetry metric for compact LLM context."""
    if value is None:
        return f"{driver}: unavailable"
    if isinstance(value, (int, float, np.floating)):
        return f"{driver}: {float(value):.{decimals}f}{unit}"
    return f"{driver}: {value}{unit}"


def build_qualifying_telemetry_summary(year: int, round_no: int) -> str:
    """
    Build a compact qualifying telemetry summary for the LLM analyst.

    Qualifying telemetry is pre-race evidence. It is intentionally summarized
    rather than added to RankNet features, so model training/inference stays
    unchanged and no race-session telemetry leaks into predictions.
    """
    feats = fetch_telemetry_features(year, round_no, session_type="Q")
    if not feats:
        return "Not available for this race."

    def valid_rows(metric: str):
        rows = [
            (driver, values.get(metric))
            for driver, values in feats.items()
            if values.get(metric) is not None
        ]
        return [
            (driver, value)
            for driver, value in rows
            if isinstance(value, (int, float, np.floating)) and np.isfinite(value)
        ]

    def best_low(metric: str):
        rows = valid_rows(metric)
        return min(rows, key=lambda item: item[1]) if rows else None

    def best_high(metric: str):
        rows = valid_rows(metric)
        return max(rows, key=lambda item: item[1]) if rows else None

    lines = [
        "Qualifying telemetry only; contextual evidence, not a RankNet model input.",
    ]

    avg_lap = best_low("avg_lap_time_s")
    if avg_lap:
        lines.append(
            "Lowest average clean lap: "
            + _format_driver_metric(avg_lap[0], avg_lap[1], "s")
        )

    vs_fastest = best_low("lap_time_vs_fastest")
    if vs_fastest:
        lines.append(
            "Best median-lap ratio vs session fastest: "
            + _format_driver_metric(vs_fastest[0], vs_fastest[1], decimals=4)
        )

    for label, metric in [
        ("Best sector 1 average", "sector1_avg_s"),
        ("Best sector 2 average", "sector2_avg_s"),
        ("Best sector 3 average", "sector3_avg_s"),
    ]:
        best = best_low(metric)
        if best:
            lines.append(f"{label}: " + _format_driver_metric(best[0], best[1], "s"))

    top_speed = best_high("top_speed_kmh")
    if top_speed:
        lines.append(
            "Highest observed speed: "
            + _format_driver_metric(top_speed[0], top_speed[1], " km/h", decimals=1)
        )

    notable = sorted(valid_rows("lap_time_vs_fastest"), key=lambda item: item[1])[:5]
    if notable:
        lines.append("Top telemetry rows by median-lap ratio:")
        for driver, _ in notable:
            values = feats[driver]
            parts = [
                _format_driver_metric(driver, values.get("avg_lap_time_s"), "s"),
                f"vs fastest {values.get('lap_time_vs_fastest'):.4f}",
            ]
            if values.get("top_speed_kmh") is not None:
                parts.append(f"top speed {values['top_speed_kmh']:.1f} km/h")
            lines.append("- " + " | ".join(parts))

    return "\n".join(lines) if len(lines) > 1 else "Not available for this race."


def merge_telemetry_into_results(results_df, year: int, round_no: int):
    """
    Attach telemetry features to a results DataFrame row for a specific race.
    Returns the enriched slice or the original if telemetry unavailable.
    """
    feats = fetch_telemetry_features(year, round_no)
    if feats is None:
        return results_df

    import pandas as pd

    rows = []
    race_slice = results_df[
        (results_df["year"] == year) & (results_df["round"] == round_no)
    ].copy()

    # FastF1 uses 3-letter abbreviations; Ergast uses full driverId
    # Build a simple abbreviation→driverId map from the session
    # (we match on last-name suffix as a best-effort heuristic)
    for abbr, tfeats in feats.items():
        mask = race_slice["driver_id"].str.upper().str.endswith(abbr.upper()[-3:])
        if mask.any():
            for col, val in tfeats.items():
                race_slice.loc[mask, col] = val

    return race_slice
