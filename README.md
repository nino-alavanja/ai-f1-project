# F1 Race Outcome Predictor

A machine-learning system that predicts the finishing order of a Formula 1 race before it happens, and lets you ask an AI analyst questions about the prediction in plain English.

Built as an individual project for the Advanced AI (BCS MBI36) course.

---

## What is Formula 1? (For non-F1 readers)

Formula 1 is a motorsport championship where 20 drivers (representing 10 teams, called *constructors*) race around a circuit roughly once every two weeks across a ~23-race season. Each race weekend has two relevant sessions:

- **Qualifying** — a time-trial session the day before the race. Each driver posts their fastest lap time; the fastest driver starts the race at the front ("pole position"), the slowest at the back. This starting grid order is a very strong predictor of the final race result.
- **The Race** — the actual competition. Drivers are awarded points (25 for 1st, 18 for 2nd, ..., 1 for 10th) that accumulate across the season into the *Driver Championship* and *Constructor Championship* standings.

The core challenge this project addresses: **can a machine learning model predict how the 20 drivers will finish, using only information available before the race starts?**

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                      Streamlit Web App (app.py)                  │
│   Predictions │ Feature Analysis │ Backtesting │ Weather │ Chat  │
└──────┬──────────────────────────────────┬────────────────────────┘
       │                                  │
┌──────▼──────────────────────┐  ┌────────▼────────────┐
│   Two-Stage PyTorch Model    │  │  OpenAI GPT-4o LLM  │
│  Stage 1: F1Net (MSE loss)   │─▶│  AI Analyst         │
│  Stage 2: RankNet (pairwise) │  │  (grounded context) │
└──────┬──────────────────────┘  └─────────────────────┘
       │ Monte-Carlo Dropout
       │ (uncertainty estimation)
┌──────▼──────────────────────────────────────────────────────────┐
│                         Data Pipeline                            │
│  Ergast F1 API ── Open-Meteo Weather API ── FastF1 Telemetry    │
│  All data cached locally in data/cache/ (CSV + JSON)            │
└─────────────────────────────────────────────────────────────────┘
```

---

## Advanced AI Techniques Used

This project combines three techniques that go beyond standard supervised learning:

### 1. Pairwise Learning-to-Rank (RankNet)
Predicting race finishing order is fundamentally a *ranking* problem, not a classification or regression problem. Standard regression (predict "finish position = 3.7") treats each driver independently and ignores the fact that only one driver can finish first, only one second, and so on.

RankNet (Burges et al., ICML 2005) solves this with a *pairwise* approach: for every pair of drivers in a race, the model learns "who finishes ahead of whom?" It trains on all possible ordered pairs (roughly 380 pairs per race, ~2.4 million total), directly optimising the ranking objective.

### 2. Monte-Carlo Dropout (Uncertainty Estimation)
A standard model gives one prediction: "Verstappen finishes 1st." A more useful model also says *how confident* it is. MC Dropout achieves this by running the same model 50 times with random neurons randomly switched off each time (a technique called Dropout). The spread of the 50 predicted rankings gives a 90% confidence interval: e.g., "Verstappen: predicted P1, but could realistically be P1–P3."

### 3. Grounded LLM Analyst
After the neural model predicts the race, a structured context — all 20 predicted drivers, all 14 feature values, rankings, win/podium probabilities, confidence intervals, and qualifying telemetry context when available — is injected into the system prompt of GPT-4o. This is a grounded structured-context analyst, not a vector-database retrieval system. GPT-4o uses the provided context to answer natural-language questions like "Why is Hamilton predicted to struggle?" or "Who benefits most from wet weather?" Qualifying telemetry is contextual evidence only and is not used as a RankNet input.

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set your OpenAI API key

Create a `.env` file in the project root:

```
OPENAI_API_KEY=sk-...
```

The key is loaded automatically. All other data (Ergast API, Open-Meteo weather) is free and needs no key. The AI Analyst tab requires the key; all other tabs work without it.

### 3. Run the app

```bash
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

### 4. Using the app (3-step workflow)

1. **Train** — Click "Train Model" in the left sidebar. The app downloads ~6 years of race history (~2,640 rows) and trains the model. Takes around 5 minutes on first run; data is cached afterwards so subsequent runs are fast.
2. **Select Race** — Choose a season (year) and a specific race from the dropdowns. The app fetches the race schedule automatically.
3. **Predict** — Click "Generate Prediction". Qualifying results are fetched automatically from the Ergast API — no manual input needed.

### 5. Train from the command line (optional)

```bash
python model/train.py
```

---

## Project Structure

```
f1_predictor/
├── app.py                  ← Web interface (5 tabs)
├── config.py               ← All hyperparameters and file paths
├── requirements.txt
├── data/
│   ├── fetch_data.py       ← Ergast API client (with pagination)
│   ├── features.py         ← Converts raw data into 14 model features
│   ├── weather.py          ← Open-Meteo weather API (33 circuits)
│   ├── telemetry.py        ← FastF1 lap-time data (optional, 2018+)
│   └── cache/              ← Auto-generated CSV/JSON cache
├── model/
│   ├── network.py          ← F1Net (regression) + RankNet (pairwise ranking)
│   ├── train.py            ← Two-stage training pipeline
│   └── predict.py          ← Inference + MC Dropout confidence intervals
└── llm/
    └── assistant.py        ← GPT-4o analyst (grounded structured context)
```

---

## Input Features (14 total)

The model uses 14 features per driver, all derived from publicly available data before the race starts:

| # | Feature | Plain-English Meaning |
|---|---|---|
| 1 | `qual_position` | Starting grid position (from qualifying) |
| 2 | `champ_points_entering` | Total championship points earned so far this season |
| 3 | `champ_position_entering` | Current position in the driver standings (1st, 2nd, ...) |
| 4 | `con_points_entering` | Team's total championship points this season |
| 5 | `con_position_entering` | Team's position in the constructor standings |
| 6 | `recent_avg_finish` | Average finishing position over the last 5 races |
| 7 | `recent_dnf_rate` | Fraction of the last 5 races where the driver retired (DNF = Did Not Finish) |
| 8 | `circuit_avg_finish` | Driver's historical average finish at this specific circuit |
| 9 | `circuit_appearances` | Number of times this driver has raced at this circuit before |
| 10 | `normalized_grid` | Grid position scaled to 0–1 (e.g. P5 of 20 = 0.25) |
| 11 | `air_temp_c` | Air temperature at race time (from Open-Meteo weather API) |
| 12 | `precipitation_mm` | Rainfall during the race hour |
| 13 | `wind_speed_kmh` | Wind speed at race time |
| 14 | `is_wet_race` | 1 if it rained, 0 if dry — rain dramatically changes race outcomes |

---

## Model Details

### Stage 1 — Regression Pre-training (F1Net)
The neural network first learns to predict exact finishing positions (1–20) using Mean Squared Error loss. This teaches the model which features matter for race pace before the ranking-specific training begins.

- Architecture: Linear(14→256) → LayerNorm → ReLU → Dropout(0.3), repeated 3 times → Linear(→1)
- Loss: Mean Squared Error on finish position
- Epochs: 150, Adam optimizer with learning-rate decay

### Stage 2 — RankNet Pairwise Fine-tuning
The pre-trained weights are transferred into a RankNet. For every race in the training data, all pairs of drivers are generated (if there are 20 drivers, that is 20×19 = 380 pairs per race). The network learns: "given driver A's features and driver B's features, which one finishes first?"

- Loss: Binary Cross-Entropy on the score difference `s(driver_A) − s(driver_B)`
- Epochs: 50, Adam lr=1e-4, L2 weight decay, gradient clipping
- At prediction time: each driver gets a single score in O(N) — no pairwise evaluation needed

### Why LayerNorm instead of BatchNorm?
The architecture uses LayerNorm (normalises within each sample) rather than the more common BatchNorm (normalises across a batch). BatchNorm maintains running statistics that behave differently during training vs inference, which caused incorrect outputs after the weight transfer between Stage 1 and Stage 2. LayerNorm has no running statistics and behaves identically in both modes.

### Uncertainty (Monte-Carlo Dropout)
At inference, Dropout is intentionally left active and the model runs 50 times per prediction. Each run produces a slightly different ranking due to the random dropout. The spread gives a 90% confidence interval per driver shown in the Predictions tab.

---

## What Each Tab Shows

| Tab | What it does |
|---|---|
| 🏆 **Predictions** | Predicted finishing order with win %, podium %, and confidence intervals |
| 📊 **Feature Analysis** | Table of all 14 features per driver; bar chart of which features matter most (permutation importance) |
| 📉 **Backtesting** | Re-predicts past races using only pre-race data, compares to actual results — shows how accurate the model is |
| 🌦 **Weather & Telemetry** | Weather conditions for the race; optional FastF1 lap-time data (available for 2018+ races only) |
| 💬 **AI Analyst** | Chat with GPT-4o about all 20 predicted drivers, uncertainty intervals, model features, and qualifying telemetry context when available |

---

## Data Sources

| Source | What it provides | Cost |
|---|---|---|
| [Ergast F1 API (jolpi.ca)](https://api.jolpi.ca/ergast/) | Race results, qualifying, standings, schedule (1950–present) | Free |
| [Open-Meteo Archive API](https://open-meteo.com/) | Historical hourly weather at 33 F1 circuits | Free |
| [FastF1](https://docs.fastf1.dev/) | Lap-time telemetry, sector splits (2018+ only) | Free |

All fetched data is cached locally in `data/cache/` so it is only downloaded once.

---

## Performance

The Backtesting tab evaluates the model by re-predicting past races using only pre-race data, then comparing predicted and actual finishing order. It now compares three approaches:

- **Qualifying-only baseline:** predicts the race result directly from qualifying order.
- **Equal-weight baseline:** equally combines normalized qualifying, driver championship strength, constructor strength, and recent form.
- **RankNet:** the learned pairwise neural ranking model.

The dashboard reports winner accuracy, top-3 hit rate, average rank error, and Spearman rank correlation for all three methods side by side.

The model reliably captures large performance gaps (dominant teams, strong qualifying-to-race correlation) but cannot predict random events like safety cars, mechanical failures, or first-corner crashes.
