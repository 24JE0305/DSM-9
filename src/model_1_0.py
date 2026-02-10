# ================================
# LEGEND HYBRID MODEL (V3.2)
# PRODUCTION SAFE | IPO SAFE
# ================================

import os
import json
import torch
import torch.nn as nn
import xgboost as xgb
import yfinance as yf
import pandas as pd
import numpy as np
from safetensors.torch import save_file

# ================================
# CONFIG
# ================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

WINDOW = 30
EPOCHS = 2000
LR = 5e-4
PATIENCE = 150

MODEL_DIR = "new_model"
os.makedirs(MODEL_DIR, exist_ok=True)

# ================================
# DATA REQUIREMENTS
# ================================

MIN_FULL_TRAIN_ROWS = 500     # LSTM + XGB
MIN_XGB_ONLY_ROWS = 250       # XGB only
MIN_WINDOW_ROWS = WINDOW + 20

# ================================
# FEATURE ENGINEERING
# ================================


def compute_rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = -delta.clip(upper=0).rolling(period).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))


def standardize(X):
    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-8
    return (X - mean) / std, mean, std


def make_sequences(X, y, window):
    Xs, ys = [], []
    for i in range(len(X) - window):
        Xs.append(X[i:i + window])
        ys.append(y[i + window])
    return np.array(Xs), np.array(ys)


def check_data_eligibility(df):
    rows = len(df)
    if rows >= MIN_FULL_TRAIN_ROWS:
        return "FULL"
    if rows >= MIN_XGB_ONLY_ROWS:
        return "XGB_ONLY"
    return "SKIP"

# ================================
# LSTM MODEL
# ================================


class LegendLSTM(nn.Module):
    def __init__(self, input_size, hidden=64, output=4):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden,
            num_layers=1,
            batch_first=True,
            dropout=0.1
        )
        self.fc = nn.Linear(hidden, output)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1])

# ================================
# MAIN TRAIN FUNCTION
# ================================


def train_legendary_hybrid(ticker):
    print(f"\n🧬 Training {ticker} (RETURN MODEL)")

    artifacts = {
        "ticker": ticker,
        "status": "FAILED",
        "lstm_path": None,
        "xgb_paths": {},
        "meta_path": None
    }

    df = yf.download(
        ticker,
        start="2019-01-01",
        end="2025-06-01",
        progress=False
    )

    if df.empty:
        print("❌ No data")
        return artifacts

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # -------- FEATURES --------
    df["RSI"] = compute_rsi(df["Close"])
    df["MA50"] = df["Close"].rolling(50).mean()
    df.dropna(inplace=True)

    # -------- RETURN TARGETS --------
    horizons = [7, 30, 90, 365]
    for h in horizons:
        df[f"R_{h}"] = df["Close"].shift(-h) / df["Close"] - 1

    df.dropna(inplace=True)

    features = ["Close", "RSI", "MA50"]
    X = df[features].values
    y = df[[f"R_{h}" for h in horizons]].values

    # -------- SCALE --------
    X, X_mean, X_std = standardize(X)

    y_mean = y.mean(axis=0)
    y_std = y.std(axis=0) + 1e-8
    y_scaled = (y - y_mean) / y_std

    # -------- SEQUENCES --------
    X_seq, y_seq = make_sequences(X, y_scaled, WINDOW)

    split = int(0.8 * len(X_seq))
    X_train = torch.tensor(X_seq[:split], dtype=torch.float32).to(DEVICE)
    y_train = torch.tensor(y_seq[:split], dtype=torch.float32).to(DEVICE)

    # -------- LSTM --------
    model = LegendLSTM(len(features)).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    loss_fn = nn.HuberLoss()

    model.train()
    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        preds = model(X_train)
        loss = loss_fn(preds, y_train)
        loss.backward()
        optimizer.step()

    # -------- SAVE LSTM --------
    lstm_path = f"{MODEL_DIR}/{ticker}_lstm.safetensors"
    save_file(model.state_dict(), lstm_path)
    artifacts["lstm_path"] = lstm_path

    # -------- XGBOOST --------
    for i, h in enumerate(horizons):
        xgb_model = xgb.XGBRegressor(
            n_estimators=500,
            max_depth=5,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            device="cuda"
        )
        xgb_model.fit(X[:-365], y_scaled[:-365, i])

        path = f"{MODEL_DIR}/{ticker}_xgb_{h}.json"
        xgb_model.save_model(path)
        artifacts["xgb_paths"][h] = path

    # -------- META --------
    meta = {
        "ticker": ticker,
        "features": features,
        "window": WINDOW,
        "X_mean": X_mean.tolist(),
        "X_std": X_std.tolist(),
        "y_mean": y_mean.tolist(),
        "y_std": y_std.tolist(),
        "horizons": horizons,
        "trained_until": str(df.index[-1].date())
    }

    meta_path = f"{MODEL_DIR}/{ticker}_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    artifacts["meta_path"] = meta_path
    artifacts["status"] = "SUCCESS"

    print(f"✅ {ticker} TRAINED")

    return artifacts
