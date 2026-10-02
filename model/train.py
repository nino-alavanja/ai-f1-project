"""
model/train.py
───────────────
Two-stage training pipeline:

  Stage 1 – Regression pre-training (F1Net, MSE loss on finish position)
  Stage 2 – RankNet fine-tuning (pairwise BCE loss over all intra-race pairs)

Saves both model weights and the scaler to disk.
Run directly:
    python model/train.py
"""

import os
import sys
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    TRAINING_YEARS, MODEL_PATH, SCALER_PATH,
    EPOCHS, BATCH_SIZE, LEARNING_RATE, HIDDEN_SIZES, DROPOUT,
    N_FEATURES,
)
from data.fetch_data import load_all_years, get_schedule_df
from data.features   import build_feature_matrix, get_X_y, FEATURE_COLS
from model.network   import F1Net, RankNet, F1Dataset, PairwiseDataset

RANKNET_EPOCHS = 50   # fine-tuning epochs on pairwise objective


def train(years: list = None, verbose: bool = True) -> dict:
    """
    Full two-stage training pipeline.

    Returns dict with training history.
    """
    if years is None:
        years = TRAINING_YEARS

    # ── 1. Load & engineer features ──────────────────────────────────────────
    if verbose:
        print("📡 Fetching F1 data …")
    raw = load_all_years(years)

    if verbose:
        print("📅 Fetching race schedule for weather data …")
    schedule_df = get_schedule_df(years)

    if verbose:
        print("🔧 Engineering features (including weather) …")
    df = build_feature_matrix(raw, schedule_df=schedule_df)
    X, y = get_X_y(df)

    if verbose:
        print(f"   Dataset size: {len(X)} rows  |  {X.shape[1]} features")

    # Guard: replace any remaining NaN/inf with column medians before scaling
    for col in range(X.shape[1]):
        mask = ~np.isfinite(X[:, col])
        if mask.any():
            finite_vals = X[~mask, col]
            X[mask, col] = float(np.median(finite_vals)) if len(finite_vals) else 0.0

    # ── 2. Fit scaler ────────────────────────────────────────────────────────
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    os.makedirs(os.path.dirname(SCALER_PATH), exist_ok=True)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)

    # ── 3. Stage 1: Regression pre-training ──────────────────────────────────
    if verbose:
        print("\n━━ Stage 1: Regression pre-training ━━")

    dataset    = F1Dataset(X, y)
    val_size   = int(0.10 * len(dataset))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reg_model = F1Net(
        input_size=X.shape[1],
        hidden_sizes=HIDDEN_SIZES,
        dropout=DROPOUT,
    ).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(reg_model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    history = {"train_loss": [], "val_mae": [], "rank_loss": []}

    if verbose:
        print(f"🏋️  Training on {device} for {EPOCHS} epochs …\n")

    for epoch in range(1, EPOCHS + 1):
        reg_model.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = reg_model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(xb)

        epoch_loss /= train_size
        history["train_loss"].append(epoch_loss)

        reg_model.eval()
        mae = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                mae += torch.abs(reg_model(xb) - yb).sum().item()
        mae /= val_size
        history["val_mae"].append(mae)
        scheduler.step()

        if verbose and (epoch % 25 == 0 or epoch == 1):
            print(f"  Epoch {epoch:3d}/{EPOCHS}  "
                  f"train_loss={epoch_loss:.4f}  val_mae={mae:.4f}")

    if verbose:
        print(f"\n✅ Stage 1 complete. Final val MAE: {mae:.4f} positions")

    # ── 4. Stage 2: RankNet fine-tuning ──────────────────────────────────────
    if verbose:
        print("\n━━ Stage 2: RankNet pairwise fine-tuning ━━")
        print("   Building pairwise dataset (all intra-race pairs) …")

    rank_model = RankNet(
        input_size=X.shape[1],
        hidden_sizes=HIDDEN_SIZES,
        dropout=DROPOUT,
    ).to(device)

    # Transfer pre-trained weights from regression backbone
    _transfer_weights(reg_model, rank_model)

    pair_ds = PairwiseDataset(df, FEATURE_COLS, scaler=scaler)
    if verbose:
        print(f"   Pairwise samples: {len(pair_ds):,}")

    pair_loader = DataLoader(pair_ds, batch_size=512, shuffle=True, drop_last=True)
    bce = nn.BCEWithLogitsLoss()
    # Low lr + weight_decay prevents BCE margin maximisation from diverging to ±inf
    rank_opt = torch.optim.Adam(
        rank_model.parameters(),
        lr=LEARNING_RATE * 0.1,   # 1e-4 — much safer than 5e-4 for fine-tuning
        weight_decay=1e-4,        # L2 regularisation keeps weights bounded
    )

    for epoch in range(1, RANKNET_EPOCHS + 1):
        rank_model.train()
        epoch_loss = 0.0
        for xi, xj, yb in pair_loader:
            xi, xj, yb = xi.to(device), xj.to(device), yb.to(device)
            rank_opt.zero_grad()
            logit_ij = rank_model(xi, xj)
            loss = bce(logit_ij, yb)
            if not torch.isfinite(loss):
                continue              # skip any corrupt batch silently
            loss.backward()
            torch.nn.utils.clip_grad_norm_(rank_model.parameters(), max_norm=1.0)
            rank_opt.step()
            epoch_loss += loss.item() * len(yb)
        epoch_loss /= len(pair_ds)
        history["rank_loss"].append(epoch_loss)
        if verbose and (epoch % 10 == 0 or epoch == 1):
            print(f"  RankNet epoch {epoch:2d}/{RANKNET_EPOCHS}  "
                  f"pair_bce={epoch_loss:.4f}")

    if verbose:
        print(f"\n✅ Stage 2 complete.")

    # ── 5. Sanity-check weights before saving ─────────────────────────────────
    import math
    for name, param in rank_model.named_parameters():
        if not torch.isfinite(param).all():
            raise RuntimeError(
                f"NaN/Inf detected in RankNet parameter '{name}' after training. "
                "This usually means a learning-rate or data issue."
            )

    # ── 6. Save RankNet (used for predictions) ────────────────────────────────
    os.makedirs(os.path.dirname(MODEL_PATH) or ".", exist_ok=True)
    torch.save(rank_model.state_dict(), MODEL_PATH)
    if verbose:
        print(f"💾 RankNet model saved to {MODEL_PATH}")

    return history


def _transfer_weights(src: F1Net, dst: RankNet):
    """
    Copy weights from the F1Net regression backbone into RankNet scorer.
    Both use LayerNorm (no running statistics), so the transfer is exact:
    we strip the 'net.' prefix from F1Net keys and prepend 'scorer.'.
    """
    src_sd = src.state_dict()
    new_sd = {}
    for k, v in src_sd.items():
        # map  net.X.weight  →  scorer.X.weight
        new_key = "scorer." + k[len("net."):]
        new_sd[new_key] = v.clone()
    dst.load_state_dict(new_sd, strict=True)


if __name__ == "__main__":
    train()
