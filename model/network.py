"""
model/network.py
─────────────────
Two architectures:

1. F1Net (regression pre-training)
   Predicts absolute finish position (1-20) per driver.

2. RankNet (pairwise ranking, fine-tuned from F1Net weights)
   For every pair of drivers in a race, predicts a logit for
   "driver A finishes ahead of driver B".
   Loss: BCEWithLogitsLoss over all intra-race pairs (Burges et al. 2005).
   Inference: rank all drivers by raw score — O(N), same cost as regression.

Architecture note: LayerNorm instead of BatchNorm1d.
  - BatchNorm depends on batch statistics in training and running stats
    in eval mode.  Running stats can drift when weights are transferred
    between models, producing extreme/NaN outputs at inference.
  - LayerNorm normalises over the feature axis within each sample; it has
    no running statistics and behaves identically in train and eval mode.
    This is the standard choice for tabular models.
"""

import torch
import torch.nn as nn


def _make_layers(input_size: int, hidden_sizes: list, dropout: float) -> nn.Sequential:
    layers = []
    prev = input_size
    for i, h in enumerate(hidden_sizes):
        layers.append(nn.Linear(prev, h))
        if i < len(hidden_sizes) - 1:
            layers.append(nn.LayerNorm(h))   # no running statistics
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))   # kept on ALL layers for MC Dropout
        prev = h
    layers.append(nn.Linear(prev, 1))
    return nn.Sequential(*layers)


class F1Net(nn.Module):
    """Regression network: predicts a real-valued finish position per driver."""

    def __init__(self, input_size: int = 14, hidden_sizes: list = None, dropout: float = 0.3):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [256, 128, 64]
        self.net = _make_layers(input_size, hidden_sizes, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class RankNet(nn.Module):
    """
    Pairwise ranking network (Burges et al., 2005).

    score(x)           → scalar relevance score per driver (used at inference)
    forward(x_i, x_j)  → raw logit s_i − s_j  (used during training with BCEWithLogitsLoss)
    """

    def __init__(self, input_size: int = 14, hidden_sizes: list = None, dropout: float = 0.3):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [256, 128, 64]
        self.scorer = _make_layers(input_size, hidden_sizes, dropout)

    def score(self, x: torch.Tensor) -> torch.Tensor:
        """Raw model score per driver. In this project, lower is ranked better."""
        return self.scorer(x).squeeze(-1)

    def forward(self, x_i: torch.Tensor, x_j: torch.Tensor) -> torch.Tensor:
        """
        Return raw logit for pair (x_i, x_j).
        Positive means i is predicted to finish ahead of j.
        Use with BCEWithLogitsLoss — do NOT apply sigmoid separately.
        """
        return self.scorer(x_i).squeeze(-1) - self.scorer(x_j).squeeze(-1)


# ── Dataset helpers ───────────────────────────────────────────────────────────

class F1Dataset(torch.utils.data.Dataset):
    """Wraps numpy arrays X, y for regression training."""

    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class PairwiseDataset(torch.utils.data.Dataset):
    """
    All intra-race ordered driver pairs for RankNet training.

    For each race we take every ordered pair (i, j) where driver i finished
    ahead of driver j and create a positive sample (label=1).
    """

    def __init__(self, df, feature_cols: list, scaler=None):
        import numpy as np

        X_i_list, X_j_list, y_list = [], [], []

        for (year, round_no), grp in df.groupby(["year", "round"]):
            grp = grp.dropna(subset=["finish_position"])
            if len(grp) < 2:
                continue

            X_race = grp[feature_cols].fillna(grp[feature_cols].median()).values.astype(np.float32)
            if scaler is not None:
                X_race = scaler.transform(X_race)
            pos = grp["finish_position"].values

            for ii in range(len(grp)):
                for jj in range(len(grp)):
                    if ii != jj and pos[ii] < pos[jj]:
                        X_i_list.append(X_race[ii])
                        X_j_list.append(X_race[jj])
                        y_list.append(1.0)

        self.X_i = torch.tensor(np.array(X_i_list), dtype=torch.float32)
        self.X_j = torch.tensor(np.array(X_j_list), dtype=torch.float32)
        self.y   = torch.tensor(np.array(y_list),   dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_i[idx], self.X_j[idx], self.y[idx]
