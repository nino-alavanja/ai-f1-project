"""
model/baseline.py
─────────────────
Transparent non-neural baselines for backtesting.
"""

import numpy as np


def _ranks_from_order(ordered_rows: list[dict]) -> dict[str, int]:
    """Return driver_id -> rank for rows already sorted best to worst."""
    return {
        row["driver_id"]: rank
        for rank, row in enumerate(ordered_rows, start=1)
    }


def rank_by_qualifying(driver_rows: list[dict]) -> dict[str, int]:
    """
    Baseline: predict race finish exactly from qualifying order.
    Lower qualifying position is better.
    """
    ordered = sorted(
        driver_rows,
        key=lambda r: (r["qual_position"], r["driver_id"]),
    )
    return _ranks_from_order(ordered)


def _normalized(values: list[float], higher_is_better: bool) -> np.ndarray:
    """
    Min-max normalize values within one race.
    Returns 0.5 for constant columns so the feature is neutral.
    """
    arr = np.asarray(values, dtype=float)
    finite = np.isfinite(arr)
    if not finite.any():
        norm = np.full(len(arr), 0.5)
    else:
        fallback = float(np.median(arr[finite]))
        arr = np.where(finite, arr, fallback)
        min_val = float(arr.min())
        max_val = float(arr.max())
        if max_val == min_val:
            norm = np.full(len(arr), 0.5)
        else:
            norm = (arr - min_val) / (max_val - min_val)

    return norm if higher_is_better else 1.0 - norm


def rank_by_equal_weight(driver_rows: list[dict]) -> dict[str, int]:
    """
    Equal-weight heuristic baseline.

    Each feature is normalized within the race, direction-adjusted so higher is
    always better, then averaged with no learned or hand-tuned coefficients.
    """
    feature_specs = [
        ("qual_position", False),
        ("champ_points_entering", True),
        ("champ_position_entering", False),
        ("con_points_entering", True),
        ("con_position_entering", False),
        ("recent_avg_finish", False),
    ]

    normalized_features = []
    for feature_name, higher_is_better in feature_specs:
        values = [
            row.get("feature_dict", {}).get(feature_name, row.get(feature_name, np.nan))
            for row in driver_rows
        ]
        normalized_features.append(_normalized(values, higher_is_better))

    scores = np.mean(np.vstack(normalized_features), axis=0)
    scored_rows = [
        {
            **row,
            "baseline_score": float(score),
        }
        for row, score in zip(driver_rows, scores)
    ]

    ordered = sorted(
        scored_rows,
        key=lambda r: (-r["baseline_score"], r["qual_position"], r["driver_id"]),
    )
    return _ranks_from_order(ordered)
