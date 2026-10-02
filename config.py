import os
from dotenv import load_dotenv
load_dotenv()

# ── OpenAI ────────────────────────────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY_HERE")

# ── Ergast F1 API (jolpi.ca mirror — free, no key needed) ────────────────────
ERGAST_BASE_URL = "https://api.jolpi.ca/ergast/f1"

# ── Data settings ────────────────────────────────────────────────────────────
TRAINING_YEARS     = list(range(2010, 2024))   # years used to train the model
RECENT_FORM_WINDOW = 5                          # races used to compute "recent form"
DATA_DIR           = "data/cache"               # where raw CSVs are saved

# ── Model settings ────────────────────────────────────────────────────────────
MODEL_PATH   = "model/f1_model.pth"
SCALER_PATH  = "model/scaler.pkl"
EPOCHS       = 150
BATCH_SIZE   = 64
LEARNING_RATE = 1e-3
HIDDEN_SIZES = [256, 128, 64]   # wider to handle 14 features
DROPOUT      = 0.3
N_FEATURES   = 14               # 10 base + 4 weather

# MC Dropout samples for confidence interval estimation
MC_DROPOUT_SAMPLES = 50

# ── LLM settings ─────────────────────────────────────────────────────────────
LLM_MODEL        = "gpt-4o"
LLM_TEMPERATURE  = 0.7
LLM_MAX_TOKENS   = 600
