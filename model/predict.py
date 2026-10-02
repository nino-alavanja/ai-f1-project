"""
model/predict.py
────────────────
Loads the trained RankNet model and generates predictions for an upcoming
race.

New in this version:
  • Uses RankNet.score() instead of F1Net regression output
  • MC Dropout: runs N forward passes with dropout *active* to estimate
    prediction uncertainty as a 90% confidence interval on finish position
  • Returns rank_std (std dev across MC samples) per driver

Main entry point: predict_race()
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    MODEL_PATH, SCALER_PATH, RECENT_FORM_WINDOW,
    HIDDEN_SIZES, DROPOUT, N_FEATURES, MC_DROPOUT_SAMPLES,
)
from model.network import RankNet
from data.features import FEATURE_COLS, build_prediction_row
from data.weather  import fetch_race_weather


def load_model(device=None):
    """Load the saved RankNet weights and scaler from disk."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"No trained model found at {MODEL_PATH}. "
            "Run model/train.py first."
        )

    model = RankNet(
        input_size=N_FEATURES,
        hidden_sizes=HIDDEN_SIZES,
        dropout=DROPOUT,
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

    return model, scaler, device


def _enable_dropout(model: torch.nn.Module):
    """Set all Dropout layers to training mode (enables MC Dropout)."""
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.train()


def compute_driver_features(
    results_df,
    driver_standings_df,
    con_standings_df,
    qualifying_results,
    circuit_id: str,
    year: int,
    round_no: int,
    race_date: str = "TBD",
) -> list[dict]:
    """
    Compute feature vectors for each driver before a race.
    Returns list of dicts with driver_id, feature_vector, feature_dict.
    """
    weather = fetch_race_weather(circuit_id, race_date)
    n_drivers = len(qualifying_results)
    driver_rows = []

    for entry in qualifying_results:
        driver_id      = entry["driver_id"]
        constructor_id = entry["constructor_id"]
        qual_pos       = entry["qual_position"]

        # Championship points/position entering this race
        drv_prev = driver_standings_df[
            (driver_standings_df["driver_id"] == driver_id) &
            (driver_standings_df["year"] == year) &
            (driver_standings_df["after_round"] < round_no)
        ].sort_values("after_round")

        champ_pts = float(drv_prev["champ_points"].iloc[-1])   if len(drv_prev) else 0.0
        champ_pos = int(drv_prev["champ_position"].iloc[-1])   if len(drv_prev) else 20

        # Constructor standings
        con_prev = con_standings_df[
            (con_standings_df["constructor_id"] == constructor_id) &
            (con_standings_df["year"] == year) &
            (con_standings_df["after_round"] < round_no)
        ].sort_values("after_round")

        con_pts = float(con_prev["con_points"].iloc[-1])   if len(con_prev) else 0.0
        con_pos = int(con_prev["con_position"].iloc[-1])   if len(con_prev) else 10

        # Recent form
        recent = results_df[
            (results_df["driver_id"] == driver_id) &
            (
                (results_df["year"] < year) |
                ((results_df["year"] == year) & (results_df["round"] < round_no))
            )
        ].sort_values(["year", "round"]).tail(RECENT_FORM_WINDOW)

        if len(recent):
            def safe_pos(row):
                if any(s.lower() in str(row["status"]).lower()
                       for s in ["accident", "engine", "retired", "mechanical",
                                  "collision", "gearbox", "hydraulics"]):
                    return 20.0
                # use pd.notna — `if NaN` is True in Python, which is wrong here
                pos = row["finish_position"]
                return float(pos) if pd.notna(pos) and pos != 0 else 20.0

            recent_finishes = recent.apply(safe_pos, axis=1).values.astype(float)
            recent_avg      = float(np.nanmean(recent_finishes))
            recent_dnf_rate = float(np.mean(recent_finishes == 20.0))
        else:
            recent_avg      = 10.0
            recent_dnf_rate = 0.0

        # Circuit history
        circuit_hist = results_df[
            (results_df["driver_id"]  == driver_id) &
            (results_df["circuit_id"] == circuit_id) &
            (
                (results_df["year"] < year) |
                ((results_df["year"] == year) & (results_df["round"] < round_no))
            )
        ]
        valid_pos   = circuit_hist["finish_position"].dropna()
        circuit_avg = float(valid_pos.mean()) if len(valid_pos) else 10.0
        circuit_app = int(len(circuit_hist))

        feature_vec = build_prediction_row(
            driver_id, constructor_id, qual_pos, n_drivers,
            champ_pts, champ_pos, con_pts, con_pos,
            recent_avg, recent_dnf_rate, circuit_avg, circuit_app,
            air_temp_c=weather["air_temp_c"],
            precipitation_mm=weather["precipitation_mm"],
            wind_speed_kmh=weather["wind_speed_kmh"],
            is_wet_race=weather["is_wet_race"],
        )

        driver_rows.append({
            "driver_id":      driver_id,
            "constructor_id": constructor_id,
            "qual_position":  qual_pos,
            "feature_vector": feature_vec,
            "feature_dict": {
                "qual_position":           qual_pos,
                "champ_points_entering":   champ_pts,
                "champ_position_entering": champ_pos,
                "con_points_entering":     con_pts,
                "con_position_entering":   con_pos,
                "recent_avg_finish":       round(recent_avg, 2),
                "recent_dnf_rate":         round(recent_dnf_rate, 2),
                "circuit_avg_finish":      round(circuit_avg, 2),
                "circuit_appearances":     circuit_app,
                "air_temp_c":              weather["air_temp_c"],
                "precipitation_mm":        weather["precipitation_mm"],
                "wind_speed_kmh":          weather["wind_speed_kmh"],
                "is_wet_race":             weather["is_wet_race"],
            },
        })

    return driver_rows


def predict_race(
    results_df,
    driver_standings_df,
    con_standings_df,
    qualifying_results,
    circuit_id: str,
    year: int,
    round_no: int,
    race_name: str = "",
    race_date: str = "TBD",
) -> list[dict]:
    """
    Main prediction function.

    Returns list[dict] sorted by predicted finish (best first), each dict:
      predicted_position, driver_id, constructor_id, qual_position,
      win_probability, podium_probability,
      rank_score, rank_ci_low, rank_ci_high,   ← NEW: confidence interval
      feature_dict
    """
    model, scaler, device = load_model()

    driver_rows = compute_driver_features(
        results_df, driver_standings_df, con_standings_df,
        qualifying_results, circuit_id, year, round_no, race_date,
    )

    X = np.stack([r["feature_vector"] for r in driver_rows])
    # Replace any NaN/inf that survived feature engineering with column medians
    # (compute medians only from finite values to avoid propagating inf)
    for col in range(X.shape[1]):
        mask = ~np.isfinite(X[:, col])
        if mask.any():
            finite_vals = X[~mask, col]
            fallback = float(np.median(finite_vals)) if len(finite_vals) else 0.0
            X[mask, col] = fallback
    X_scaled = scaler.transform(X).astype(np.float32)
    X_tensor = torch.tensor(X_scaled, device=device)

    # ── Point estimate (eval mode, no dropout) ───────────────────────────────
    model.eval()
    with torch.no_grad():
        scores = model.score(X_tensor).cpu().numpy()

    # ── MC Dropout: N stochastic forward passes ──────────────────────────────
    _enable_dropout(model)
    mc_scores = np.zeros((MC_DROPOUT_SAMPLES, len(driver_rows)))
    with torch.no_grad():
        for s in range(MC_DROPOUT_SAMPLES):
            mc_scores[s] = model.score(X_tensor).cpu().numpy()

    # Convert score samples → rank samples (1 = best).
    # Lower scores are better for the saved RankNet because it was initialized
    # from finish-position regression, where P1 is lower than P20.
    mc_ranks = np.zeros_like(mc_scores)
    for s in range(MC_DROPOUT_SAMPLES):
        order = np.argsort(mc_scores[s])
        for rank, idx in enumerate(order, start=1):
            mc_ranks[s, idx] = rank

    rank_mean = mc_ranks.mean(axis=0)
    rank_std  = mc_ranks.std(axis=0)
    rank_ci_low  = np.percentile(mc_ranks, 5,  axis=0)
    rank_ci_high = np.percentile(mc_ranks, 95, axis=0)

    # ── Final ranking (by point-estimate score, lower = better) ──────────────
    order = np.argsort(scores)

    # Win/podium probabilities via softmax over inverted scores
    # because lower RankNet scores indicate stronger predicted finish.
    probability_scores = -scores
    exp_scores = np.exp(probability_scores - probability_scores.max())
    win_probs  = exp_scores / exp_scores.sum()

    top3_mask    = np.zeros(len(scores))
    top3_mask[order[:3]] = 1
    podium_probs = np.where(top3_mask, win_probs * 3, win_probs * 0.5)
    podium_probs = np.clip(podium_probs, 0, 1)

    results = []
    for rank, idx in enumerate(order, start=1):
        row = driver_rows[idx]
        results.append({
            "predicted_position": rank,
            "driver_id":          row["driver_id"],
            "constructor_id":     row["constructor_id"],
            "qual_position":      row["qual_position"],
            "rank_score":         float(scores[idx]),
            "win_probability":    float(win_probs[idx]),
            "podium_probability": float(podium_probs[idx]),
            "rank_mean":          float(rank_mean[idx]),
            "rank_std":           float(rank_std[idx]),
            "rank_ci_low":        float(rank_ci_low[idx]),
            "rank_ci_high":       float(rank_ci_high[idx]),
            "feature_dict":       row["feature_dict"],
        })

    return results
