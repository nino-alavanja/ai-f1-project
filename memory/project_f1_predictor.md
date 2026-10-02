---
name: F1 Predictor project state
description: Architecture, fixes applied, and features implemented for the F1 Race Predictor advanced AI project
type: project
---

Advanced AI course project: F1 race outcome predictor with LLM analyst.

**Why:** ~60-hour individual project using advanced ML (RankNet + LLM)

**How to apply:** Reference when resuming work; model must be retrained after code changes

## Bugs fixed (2025-05-09)
- `.env` had `OPEN_API_KEY` (typo) → fixed to `OPENAI_API_KEY`
- API pagination: jolpi.ca Ergast mirror caps at 100 results regardless of `limit=` param; fixed with `_get_all_pages()` helper using offset pagination — now fetches all rounds (was only 5/22 per year)
- Stale cache deleted; needs re-download on first run

## Architecture
- `data/fetch_data.py` – Ergast API client with pagination fix + `get_schedule_df()`
- `data/weather.py` – Open-Meteo historical weather (free, no key) for 33 circuits
- `data/features.py` – 14-feature engineering (10 base + 4 weather)
- `data/telemetry.py` – FastF1 lap-time telemetry (2018+), optional enrichment
- `model/network.py` – F1Net (regression) + RankNet (pairwise) + PairwiseDataset
- `model/train.py` – Two-stage: MSE pre-train → RankNet BCE fine-tune
- `model/predict.py` – RankNet inference + MC Dropout (50 passes) for 90% CI
- `llm/assistant.py` – GPT-4o analyst grounded in predictions
- `app.py` – Streamlit with 5 tabs: Predictions, Feature Analysis, Backtesting, Weather/Telemetry, AI Chat

## Features (14)
qual_position, champ_points_entering, champ_position_entering, con_points_entering, con_position_entering, recent_avg_finish, recent_dnf_rate, circuit_avg_finish, circuit_appearances, normalized_grid, air_temp_c, precipitation_mm, wind_speed_kmh, is_wet_race

## Key configs
- `config.py`: MC_DROPOUT_SAMPLES=50, N_FEATURES=14, HIDDEN_SIZES=[256,128,64]
- Training: EPOCHS=150 regression + RANKNET_EPOCHS=30 pairwise
- Model saved as RankNet at `model/f1_model.pth`

## Run
```
pip install -r requirements.txt
streamlit run app.py
```
