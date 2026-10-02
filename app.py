"""
app.py
──────
Streamlit application for the F1 Race Predictor.

Tabs:
  🏆 Predictions          – predicted order, win %, podium %, confidence intervals
  📊 Feature Analysis     – per-driver feature table + feature importance (permutation)
  📉 Backtesting          – compare past predictions vs actual results
  🌦 Weather & Telemetry  – weather at race + FastF1 lap-time panel
  💬 AI Analyst           – GPT-4o chat grounded in the predictions
"""

import os
import sys
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from config import TRAINING_YEARS, MODEL_PATH, N_FEATURES
from data.fetch_data import (
    load_all_years, get_race_schedule, fetch_upcoming_race_info,
    fetch_race_results, fetch_driver_standings_by_round,
    fetch_constructor_standings_by_round, fetch_qualifying,
    get_schedule_df,
)
from data.features   import build_feature_matrix, FEATURE_COLS
from data.weather    import fetch_race_weather
from data.telemetry  import fetch_telemetry_features, build_qualifying_telemetry_summary
from model.train     import train as train_model
from model.predict   import predict_race, load_model, compute_driver_features
from model.baseline  import rank_by_qualifying, rank_by_equal_weight
from llm.assistant   import F1Assistant, build_race_context

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="F1 Race Predictor",
    page_icon="🏎️",
    layout="wide",
)

# ── Session state defaults ────────────────────────────────────────────────────
def _init_state():
    defaults = {
        "model_trained":  os.path.exists(MODEL_PATH),
        "predictions":    None,
        "race_info":      None,
        "assistant":      None,
        "chat_history":   [],
        "raw_data":       None,
        "pred_year":      2024,
        "pred_round":     1,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS  (must be defined before the UI code that calls them)
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_feature_importance(
    preds, all_results, drv_st, con_st,
    race_info, race_date, qualifying_results,
) -> dict:
    """Permutation importance: shuffle each feature, measure change in predicted ranking."""
    try:
        import torch
        from model.predict import load_model
        model, scaler, device = load_model()
        circuit_id = race_info.get("circuit_id", "unknown")
        year     = st.session_state.pred_year
        round_no = st.session_state.pred_round

        driver_rows = compute_driver_features(
            all_results, drv_st, con_st, qualifying_results,
            circuit_id, year, round_no, race_date,
        )
        X = np.stack([r["feature_vector"] for r in driver_rows])
        for col in range(X.shape[1]):
            mask = ~np.isfinite(X[:, col])
            if mask.any():
                finite_vals = X[~mask, col]
                X[mask, col] = float(np.median(finite_vals)) if len(finite_vals) else 0.0
        X_scaled = scaler.transform(X).astype(np.float32)

        model.eval()
        with torch.no_grad():
            base_scores = model.score(torch.tensor(X_scaled, device=device)).cpu().numpy()
        base_order = np.argsort(-base_scores)

        importance = {}
        rng = np.random.default_rng(42)
        for i, feat_name in enumerate(FEATURE_COLS):
            X_perm = X_scaled.copy()
            X_perm[:, i] = rng.permutation(X_perm[:, i])
            with torch.no_grad():
                perm_scores = model.score(torch.tensor(X_perm, device=device)).cpu().numpy()
            perm_order = np.argsort(-perm_scores)
            change = float(np.mean(np.abs(np.argsort(base_order) - np.argsort(perm_order))))
            importance[feat_name] = round(change, 3)

        return importance
    except Exception:
        return {}


def _evaluate_ranking(pred_ranks: dict, actual_ranks: dict) -> dict:
    """Compute common backtest metrics for one predicted ranking."""
    from scipy.stats import spearmanr

    common = [d for d in pred_ranks if d in actual_ranks]
    if not common:
        return {
            "top1_correct": 0,
            "top3_acc": 0.0,
            "mae": np.nan,
            "spearman": np.nan,
        }

    predicted_winner = min(pred_ranks, key=pred_ranks.get)
    actual_winner = min(actual_ranks, key=actual_ranks.get)
    pred_top3 = {d for d, r in pred_ranks.items() if r <= 3}
    actual_top3 = {d for d, r in actual_ranks.items() if r <= 3}
    top3_denominator = min(3, len(actual_top3)) or 1

    p_ranks = [pred_ranks[d] for d in common]
    a_ranks = [actual_ranks[d] for d in common]
    corr, _ = spearmanr(p_ranks, a_ranks) if len(common) > 1 else (np.nan, None)

    return {
        "top1_correct": int(predicted_winner == actual_winner),
        "top3_acc": len(pred_top3 & actual_top3) / top3_denominator,
        "mae": float(np.mean([abs(pred_ranks[d] - actual_ranks[d]) for d in common])),
        "spearman": float(corr) if np.isfinite(corr) else np.nan,
    }


def _run_backtest(year: int, n_races: int) -> list[dict]:
    """
    For each of the first n_races in year: predict using pre-race data,
    compare to actual result, compute accuracy metrics.
    """
    import torch

    results_out = []
    try:
        schedule = get_race_schedule(year)[:n_races]
        years_needed = list(range(max(2010, year - 5), year + 1))

        all_results = pd.concat(
            [fetch_race_results(y) for y in years_needed], ignore_index=True,
        )
        drv_st = pd.concat(
            [fetch_driver_standings_by_round(y) for y in years_needed], ignore_index=True,
        )
        con_st = pd.concat(
            [fetch_constructor_standings_by_round(y) for y in years_needed], ignore_index=True,
        )

        for race in schedule:
            round_no   = race["round"]
            race_name  = race["race_name"]
            circuit_id = race["circuit_id"]
            race_date  = race.get("date", "TBD")

            try:
                qual_df = fetch_qualifying(year)
                q_race  = qual_df[qual_df["round"] == round_no]
            except Exception:
                continue
            if q_race.empty:
                continue

            res_race = all_results[
                (all_results["year"] == year) & (all_results["round"] == round_no)
            ]
            if res_race.empty:
                continue

            drv_cons = res_race.set_index("driver_id")["constructor_id"].to_dict()
            quali_entries = [
                {
                    "driver_id":      row["driver_id"],
                    "constructor_id": drv_cons.get(row["driver_id"], "unknown"),
                    "qual_position":  int(row["qual_position"]),
                }
                for _, row in q_race.iterrows()
            ]

            try:
                from model.predict import load_model
                model, scaler, device = load_model()
                driver_rows = compute_driver_features(
                    all_results, drv_st, con_st,
                    quali_entries, circuit_id, year, round_no, race_date,
                )
                X = np.stack([r["feature_vector"] for r in driver_rows])
                for col in range(X.shape[1]):
                    mask = ~np.isfinite(X[:, col])
                    if mask.any():
                        finite_vals = X[~mask, col]
                        X[mask, col] = float(np.median(finite_vals)) if len(finite_vals) else 0.0
                X_sc = scaler.transform(X).astype(np.float32)
                model.eval()
                with torch.no_grad():
                    scores = model.score(torch.tensor(X_sc, device=device)).cpu().numpy()
                pred_order = np.argsort(scores)
                ranknet_ranks = {
                    driver_rows[idx]["driver_id"]: rank + 1
                    for rank, idx in enumerate(pred_order)
                }
                qualifying_ranks = rank_by_qualifying(driver_rows)
                equal_weight_ranks = rank_by_equal_weight(driver_rows)
            except Exception:
                continue

            actual = res_race[["driver_id", "finish_position"]].dropna()
            actual_ranks = dict(zip(actual["driver_id"], actual["finish_position"].astype(int)))
            common = [d for d in ranknet_ranks if d in actual_ranks]
            if not common:
                continue

            metrics = {
                "RankNet": _evaluate_ranking(ranknet_ranks, actual_ranks),
                "Qualifying Only": _evaluate_ranking(qualifying_ranks, actual_ranks),
                "Equal Weight Baseline": _evaluate_ranking(equal_weight_ranks, actual_ranks),
            }

            comparison = [
                {
                    "Driver":             d.replace("_", " ").title(),
                    "Actual":             actual_ranks.get(d, "–"),
                    "RankNet":            ranknet_ranks.get(d, "–"),
                    "Qualifying Only":    qualifying_ranks.get(d, "–"),
                    "Equal Weight":       equal_weight_ranks.get(d, "–"),
                    "RankNet Delta":      abs(actual_ranks[d] - ranknet_ranks[d])
                                          if d in ranknet_ranks else "–",
                    "Qualifying Delta":   abs(actual_ranks[d] - qualifying_ranks[d])
                                          if d in qualifying_ranks else "–",
                    "Equal Weight Delta": abs(actual_ranks[d] - equal_weight_ranks[d])
                                          if d in equal_weight_ranks else "–",
                }
                for d in sorted(actual_ranks, key=actual_ranks.get)
            ]

            results_out.append({
                "round":      round_no,
                "race_name":  race_name,
                "metrics":    metrics,
                "comparison": comparison,
            })

    except Exception as e:
        st.error(f"Backtest error: {e}")

    return results_out


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.title("🏎️ F1 Race Predictor")
    st.markdown("---")

    # ── Train model ───────────────────────────────────────────────────────────
    st.subheader("1. Train Model")
    year_range = st.slider("Training years", min_value=2010, max_value=2023, value=(2018, 2023))
    selected_years = list(range(year_range[0], year_range[1] + 1))

    if st.button("🏋️ Train Model", use_container_width=True):
        with st.spinner("Fetching data and training … (this may take several minutes)"):
            history = train_model(years=selected_years, verbose=False)
            st.session_state.model_trained = True
        st.success(
            f"✅ Model trained!\n"
            f"Regression val MAE: {history['val_mae'][-1]:.2f} pos | "
            f"RankNet BCE: {history['rank_loss'][-1]:.4f}"
        )

    if st.session_state.model_trained:
        st.success("✅ Model ready")
    else:
        st.warning("⚠️ No trained model yet. Click Train Model.")

    st.markdown("---")

    # ── Select race ───────────────────────────────────────────────────────────
    st.subheader("2. Select Race to Predict")
    pred_year = st.selectbox("Season", list(range(2026, 2014, -1)), index=0)
    st.session_state.pred_year = pred_year

    schedule = []
    try:
        schedule = get_race_schedule(pred_year)
    except Exception:
        st.warning("Could not load schedule. Check your connection.")

    race_options = {f"R{r['round']:02d} – {r['race_name']}": r for r in schedule}
    selected_race_label = st.selectbox("Race", list(race_options.keys()))
    selected_race_info  = race_options.get(selected_race_label, {})
    selected_round      = selected_race_info.get("round", 1)
    st.session_state.pred_round = selected_round

    st.markdown("---")

    # ── Predict button ────────────────────────────────────────────────────────
    st.subheader("3. Generate Prediction")
    st.caption("Qualifying results are fetched automatically from the Ergast API.")

    if st.button("🔮 Generate Prediction", use_container_width=True,
                 disabled=not st.session_state.model_trained):
        with st.spinner("Loading data and running model …"):
            try:
                years_needed = list(range(max(2010, pred_year - 5), pred_year + 1))
                all_results  = pd.concat(
                    [fetch_race_results(y) for y in years_needed if y <= pred_year],
                    ignore_index=True,
                )
                drv_st = pd.concat(
                    [fetch_driver_standings_by_round(y) for y in years_needed if y <= pred_year],
                    ignore_index=True,
                )
                con_st = pd.concat(
                    [fetch_constructor_standings_by_round(y) for y in years_needed if y <= pred_year],
                    ignore_index=True,
                )

                race_info  = fetch_upcoming_race_info(pred_year, selected_round)
                circuit_id = race_info.get("circuit_id", "unknown")
                race_date  = selected_race_info.get("date", "TBD")

                # Auto-fetch qualifying from Ergast
                qual_df = fetch_qualifying(pred_year)
                q_race  = qual_df[qual_df["round"] == selected_round]
                if q_race.empty:
                    st.error(
                        "No qualifying data found for this race on the Ergast API. "
                        "The session may not have happened yet."
                    )
                    st.stop()
                qualifying_results = q_race[["driver_id", "constructor_id", "qual_position"]].to_dict("records")

                predictions = predict_race(
                    results_df=all_results,
                    driver_standings_df=drv_st,
                    con_standings_df=con_st,
                    qualifying_results=qualifying_results,
                    circuit_id=circuit_id,
                    year=pred_year,
                    round_no=selected_round,
                    race_name=race_info.get("race_name", ""),
                    race_date=race_date,
                )

                telemetry_context = build_qualifying_telemetry_summary(
                    pred_year,
                    selected_round,
                )

                race_context = build_race_context(
                    race_name=race_info.get("race_name", "Unknown Race"),
                    circuit_name=race_info.get("circuit_name", "Unknown Circuit"),
                    country=race_info.get("country", ""),
                    date=race_date,
                    predictions=predictions,
                    telemetry_context=telemetry_context,
                )
                st.session_state.predictions  = predictions
                st.session_state.race_info    = race_info
                st.session_state.race_date    = race_date
                st.session_state.assistant_context_label = (
                    f"{pred_year} {race_info.get('race_name', 'Unknown Race')}"
                )
                st.session_state.qualifying_results = qualifying_results
                st.session_state.telemetry_context = telemetry_context
                st.session_state.assistant    = F1Assistant(race_context)
                st.session_state.chat_history = []
                st.session_state.all_results  = all_results
                st.session_state.drv_st       = drv_st
                st.session_state.con_st       = con_st

                opening = st.session_state.assistant.get_opening_analysis()
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": opening}
                )

            except Exception as e:
                st.error(f"Prediction failed: {e}")
                raise

    st.markdown("---")
    st.caption("Data: Ergast API (jolpi.ca) + Open-Meteo + FastF1 | Model: RankNet (PyTorch) | LLM: GPT-4o")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN PANEL
# ═══════════════════════════════════════════════════════════════════════════════

if st.session_state.predictions is None:
    st.markdown(
        """
        # 🏎️ F1 Race Outcome Predictor

        **How to use:**
        1. **Train** the neural network using the sidebar (first run: ~5 min to download data)
        2. **Select** a race season and round
        3. **Enter** qualifying results (or use the defaults)
        4. Click **Generate Prediction**
        5. Explore tabs: predictions, feature importance, backtesting, weather, AI chat

        ---
        **Model architecture:** Two-stage training —
        1. Regression pre-training (MSE on finish position)
        2. **RankNet** pairwise fine-tuning (binary cross-entropy on all driver pairs)

        **Uncertainty:** Monte-Carlo Dropout (50 stochastic passes → 90 % confidence interval)

        **Features (14):** qualifying position · championship standings ·
        constructor strength · recent form · DNF rate · circuit history ·
        air temperature · precipitation · wind speed · wet-race flag
        """
    )
else:
    preds     = st.session_state.predictions
    race_info = st.session_state.race_info
    race_date = st.session_state.get("race_date", "TBD")

    st.title(f"🏁 {race_info.get('race_name', 'F1 Race')} – {st.session_state.pred_year}")
    st.caption(
        f"📍 {race_info.get('circuit_name', '')} | "
        f"🗓 {race_date}"
    )

    tab_pred, tab_feat, tab_back, tab_wx, tab_chat = st.tabs([
        "🏆 Predictions",
        "📊 Feature Analysis",
        "📉 Backtesting",
        "🌦 Weather & Telemetry",
        "💬 AI Analyst",
    ])

    # ── Tab 1: Predictions ────────────────────────────────────────────────────
    with tab_pred:
        col_table, col_chart = st.columns([1, 1])

        with col_table:
            st.subheader("Predicted Finishing Order")
            table_data = []
            medals = ["🥇", "🥈", "🥉"]
            for p in preds:
                pos   = p["predicted_position"]
                medal = medals[pos - 1] if pos <= 3 else f"P{pos}"
                table_data.append({
                    "Pos":      medal,
                    "Driver":   p["driver_id"].replace("_", " ").title(),
                    "Team":     p["constructor_id"].replace("_", " ").title(),
                    "Grid":     f"P{p['qual_position']}",
                    "Win %":    f"{p['win_probability']*100:.1f}%",
                    "Podium %": f"{p['podium_probability']*100:.1f}%",
                    "CI 90%":   f"P{p['rank_ci_low']:.0f}–P{p['rank_ci_high']:.0f}",
                })
            st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)

        with col_chart:
            st.subheader("Win Probability + Confidence Interval")
            top8 = preds[:8]
            drivers_top8 = [p["driver_id"].replace("_", " ").title() for p in top8]
            win_pcts     = [p["win_probability"] * 100 for p in top8]
            ci_low       = [p["rank_ci_low"]  for p in top8]
            ci_high      = [p["rank_ci_high"] for p in top8]

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=drivers_top8, y=win_pcts,
                marker_color="#e10600",
                name="Win probability",
            ))
            # Confidence interval as error bars on predicted position axis
            fig.update_layout(
                yaxis_title="Win Probability (%)",
                showlegend=False,
                margin=dict(t=20, b=60),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig, use_container_width=True)

        # Confidence interval chart (predicted rank range)
        st.subheader("Predicted Rank ± 90% Confidence Interval")
        ci_fig = go.Figure()
        for p in preds[:10]:
            drv = p["driver_id"].replace("_", " ").title()
            ci_fig.add_trace(go.Scatter(
                x=[drv, drv],
                y=[p["rank_ci_low"], p["rank_ci_high"]],
                mode="lines",
                line=dict(color="#aaaaaa", width=3),
                showlegend=False,
            ))
            ci_fig.add_trace(go.Scatter(
                x=[drv],
                y=[p["rank_mean"]],
                mode="markers",
                marker=dict(color="#e10600", size=10),
                showlegend=False,
            ))
        ci_fig.update_layout(
            yaxis_title="Predicted Finish Position",
            yaxis_autorange="reversed",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=20, b=60),
        )
        st.plotly_chart(ci_fig, use_container_width=True)

    # ── Tab 2: Feature Analysis ───────────────────────────────────────────────
    with tab_feat:
        st.subheader("Feature Values Per Driver")
        rows = []
        for p in preds:
            fd = p["feature_dict"]
            rows.append({
                "Driver":        p["driver_id"].replace("_", " ").title(),
                "Quali Pos":     fd["qual_position"],
                "Champ Pts":     fd["champ_points_entering"],
                "Champ Pos":     fd["champ_position_entering"],
                "Team Pts":      fd["con_points_entering"],
                "Recent Avg":    fd["recent_avg_finish"],
                "DNF Rate":      f"{fd['recent_dnf_rate']*100:.0f}%",
                "Circuit Avg":   fd["circuit_avg_finish"],
                "Circuit Races": fd["circuit_appearances"],
                "Temp °C":       fd.get("air_temp_c", "–"),
                "Rain mm":       fd.get("precipitation_mm", "–"),
                "Wind km/h":     fd.get("wind_speed_kmh", "–"),
                "Wet":           "🌧" if fd.get("is_wet_race") else "☀️",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        st.markdown("---")
        st.subheader("Recent Form vs. Circuit History")
        fig2 = go.Figure()
        drivers   = [p["driver_id"].replace("_", " ").title() for p in preds[:10]]
        recent    = [p["feature_dict"]["recent_avg_finish"] for p in preds[:10]]
        circuit   = [p["feature_dict"]["circuit_avg_finish"] for p in preds[:10]]
        fig2.add_trace(go.Bar(name="Recent avg finish",  x=drivers, y=recent,  marker_color="#e10600"))
        fig2.add_trace(go.Bar(name="Circuit avg finish", x=drivers, y=circuit, marker_color="#1e1e1e"))
        fig2.update_layout(
            barmode="group",
            yaxis_title="Avg Finish Position (lower = better)",
            yaxis_autorange="reversed",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=20, b=60),
        )
        st.plotly_chart(fig2, use_container_width=True)

        # ── Permutation feature importance ────────────────────────────────────
        st.markdown("---")
        st.subheader("Feature Importance (Permutation)")
        st.caption(
            "Each bar shows how much the ranking changes when that feature is shuffled. "
            "Larger = more important."
        )
        if st.button("Compute Feature Importance", key="fi_btn"):
            with st.spinner("Running permutation importance …"):
                importance = _compute_feature_importance(
                    preds, st.session_state.all_results,
                    st.session_state.drv_st, st.session_state.con_st,
                    race_info, race_date, st.session_state.qualifying_results,
                )
            if importance:
                fi_df = pd.DataFrame(
                    sorted(importance.items(), key=lambda x: -x[1]),
                    columns=["Feature", "Importance (rank change)"],
                )
                fi_fig = px.bar(
                    fi_df, x="Importance (rank change)", y="Feature",
                    orientation="h", color="Importance (rank change)",
                    color_continuous_scale="reds",
                )
                fi_fig.update_layout(
                    showlegend=False, coloraxis_showscale=False,
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    margin=dict(t=20),
                )
                st.plotly_chart(fi_fig, use_container_width=True)

    # ── Tab 3: Backtesting ────────────────────────────────────────────────────
    with tab_back:
        st.subheader("📉 Backtesting – Predictions vs. Actual Results")
        st.caption(
            "Select a past season to retrodict. For each race we predict using only "
            "data available *before* that race, then compare against actual results."
        )

        bt_col1, bt_col2 = st.columns([1, 2])
        with bt_col1:
            bt_year  = st.selectbox("Season to backtest", list(range(2023, 2018, -1)), key="bt_year")
            bt_races = st.slider("Number of races to test", 1, 10, 5, key="bt_races")
            run_bt   = st.button("▶ Run Backtest", key="bt_run")

        if run_bt:
            with st.spinner(f"Running backtest on {bt_year} (first {bt_races} races) …"):
                bt_results = _run_backtest(bt_year, bt_races)
                st.session_state["bt_results"] = bt_results

        if "bt_results" in st.session_state and st.session_state["bt_results"]:
            bt_data = st.session_state["bt_results"]

            model_names = list(bt_data[0]["metrics"].keys())
            summary_rows = []
            for model_name in model_names:
                model_metrics = [r["metrics"][model_name] for r in bt_data]
                summary_rows.append({
                    "Model": model_name,
                    "Winner Correct": f"{np.nanmean([m['top1_correct'] for m in model_metrics]) * 100:.0f}%",
                    "Top-3 Hit Rate": f"{np.nanmean([m['top3_acc'] for m in model_metrics]) * 100:.0f}%",
                    "Avg Rank Error": f"{np.nanmean([m['mae'] for m in model_metrics]):.1f}",
                    "Spearman ρ": f"{np.nanmean([m['spearman'] for m in model_metrics]):.2f}",
                })

            st.dataframe(
                pd.DataFrame(summary_rows),
                use_container_width=True,
                hide_index=True,
            )

            # Per-race detail
            st.markdown("---")
            for r in bt_data:
                with st.expander(f"Round {r['round']} — {r['race_name']}"):
                    comp_df = pd.DataFrame(r["comparison"])
                    st.dataframe(comp_df, use_container_width=True, hide_index=True)

    # ── Tab 4: Weather & Telemetry ────────────────────────────────────────────
    with tab_wx:
        circuit_id = race_info.get("circuit_id", "unknown")
        weather    = fetch_race_weather(circuit_id, race_date)

        st.subheader("🌦 Race-Day Weather")
        wc1, wc2, wc3, wc4 = st.columns(4)
        wc1.metric("Air Temp", f"{weather['air_temp_c']} °C")
        wc2.metric("Precipitation", f"{weather['precipitation_mm']} mm")
        wc3.metric("Wind Speed", f"{weather['wind_speed_kmh']} km/h")
        wc4.metric("Condition", "🌧 WET" if weather["is_wet_race"] else "☀️ DRY")

        st.markdown("---")
        st.subheader("🏎️ FastF1 Lap-Time Telemetry")
        st.caption(
            f"Fetches actual lap data for this race if it has already been run "
            f"(FastF1 data available from 2018+). This may take a moment."
        )

        if st.button("Load Telemetry Data", key="telem_btn"):
            with st.spinner("Loading FastF1 session …"):
                telem = fetch_telemetry_features(
                    st.session_state.pred_year,
                    st.session_state.pred_round,
                )
            if telem:
                telem_rows = []
                for drv_abbr, feats in telem.items():
                    telem_rows.append({
                        "Driver":         drv_abbr,
                        "Avg Lap (s)":    feats.get("avg_lap_time_s"),
                        "vs Fastest":     feats.get("lap_time_vs_fastest"),
                        "Top Speed":      feats.get("top_speed_kmh"),
                        "S1 Avg (s)":     feats.get("sector1_avg_s"),
                        "S2 Avg (s)":     feats.get("sector2_avg_s"),
                        "S3 Avg (s)":     feats.get("sector3_avg_s"),
                        "Tyre Stints":    feats.get("tyre_stints"),
                        "Top Compound":   feats.get("most_used_compound"),
                    })
                st.dataframe(pd.DataFrame(telem_rows), use_container_width=True, hide_index=True)

                # Lap time vs fastest scatter
                valid = [(r["Driver"], r["vs Fastest"]) for r in telem_rows
                         if r["vs Fastest"] is not None]
                if valid:
                    vdf = pd.DataFrame(valid, columns=["Driver", "Median lap vs fastest"])
                    vdf = vdf.sort_values("Median lap vs fastest")
                    fig_telem = px.bar(
                        vdf, x="Driver", y="Median lap vs fastest",
                        color="Median lap vs fastest",
                        color_continuous_scale="reds_r",
                        title="Median lap time relative to session fastest (1.00 = matched fastest)",
                    )
                    fig_telem.update_layout(
                        coloraxis_showscale=False,
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                    )
                    st.plotly_chart(fig_telem, use_container_width=True)
            else:
                st.info(
                    "No telemetry data available for this race yet "
                    "(race may not have occurred, or FastF1 doesn't have data for this round)."
                )

    # ── Tab 5: AI Analyst Chat ────────────────────────────────────────────────
    with tab_chat:
        st.subheader("💬 Chat with the F1 AI Analyst")
        st.caption(
            "Ask anything about the predictions — strategy, why a driver is "
            "favoured, circuit characteristics, upset potential, weather impact, etc."
        )
        if "assistant_context_label" in st.session_state:
            st.caption(f"Current assistant context: {st.session_state.assistant_context_label}")
        st.caption(
            "When available, the assistant also receives qualifying telemetry "
            "as contextual evidence. Telemetry is not used as a RankNet input."
        )

        for msg in st.session_state.chat_history:
            role   = msg["role"]
            avatar = "🤖" if role == "assistant" else "👤"
            with st.chat_message(role, avatar=avatar):
                st.write(msg["content"])

        user_input = st.chat_input("Ask me about this race prediction …")
        if user_input:
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            with st.chat_message("user", avatar="👤"):
                st.write(user_input)
            with st.chat_message("assistant", avatar="🤖"):
                with st.spinner("Thinking …"):
                    response = st.session_state.assistant.ask(user_input)
                st.write(response)
            st.session_state.chat_history.append({"role": "assistant", "content": response})
