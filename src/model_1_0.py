# ================================
# LEGEND HYBRID MODEL (V3.3)
# ROLLING 5Y + INCREMENTAL RETRAIN
# ================================

import os
import json
import datetime
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
EPOCHS = 1500
LR = 5e-4

MIN_NEW_DAYS = 14
MIN_FULL_TRAIN_ROWS = 500
MIN_XGB_ONLY_ROWS = 250

# ================================
# UTILS
# ================================


def today():
    return datetime.date.today()


def five_years_ago():
    return today() - datetime.timedelta(days=365 * 5)


def yesterday():
    return today() - datetime.timedelta(days=1)

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


def check_mode(df):
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
    def __init__(self, input_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, 64, batch_first=True)
        self.fc = nn.Linear(64, 4)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1])

# ================================
# MAIN TRAIN / UPDATE FUNCTION
# ================================


def train_legendary_hybrid(
    ticker: str,
    output_dir: str,
    mode: str = "fresh"   # fresh | update
):
    os.makedirs(output_dir, exist_ok=True)

    meta_path = f"{output_dir}/meta.json"

    # -------- DATE LOGIC --------
    start_date = five_years_ago()
    end_date = yesterday()

    if mode == "update" and os.path.exists(meta_path):
        with open(meta_path) as f:
            old_meta = json.load(f)

        trained_until = datetime.date.fromisoformat(old_meta["trained_until"])
        delta_days = (end_date - trained_until).days

        if delta_days < MIN_NEW_DAYS:
            print(f"⏩ {ticker}: Only {delta_days} new days — skipping update")
            return {"ticker": ticker, "status": "SKIPPED"}

        start_date = trained_until + datetime.timedelta(days=1)

    print(f"🧬 {ticker} | {mode.upper()} | {start_date} → {end_date}")

    # -------- DOWNLOAD DATA --------
    df = yf.download(
        ticker,
        start=start_date.isoformat(),
        end=end_date.isoformat(),
        progress=False
    )

    if df.empty:
        print("❌ No data")
        return {"ticker": ticker, "status": "FAILED"}

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # -------- FEATURES --------
    df["RSI"] = compute_rsi(df["Close"])
    df["MA50"] = df["Close"].rolling(50).mean()
    df.dropna(inplace=True)

    mode_flag = check_mode(df)
    if mode_flag == "SKIP":
        print("⚠️ Not enough data")
        return {"ticker": ticker, "status": "SKIPPED"}

    # -------- TARGETS --------
    horizons = [7, 30, 90, 365]
    for h in horizons:
        df[f"R_{h}"] = df["Close"].shift(-h) / df["Close"] - 1

    df.dropna(inplace=True)

    features = ["Close", "RSI", "MA50"]
    X = df[features].values
    y = df[[f"R_{h}" for h in horizons]].values

    X, X_mean, X_std = standardize(X)
    y_mean, y_std = y.mean(0), y.std(0) + 1e-8
    y_scaled = (y - y_mean) / y_std

    # -------- XGB --------
    for i, h in enumerate(horizons):
        model = xgb.XGBRegressor(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.03,
            tree_method="hist",
            device="cuda"
        )
        model.fit(X[:-365], y_scaled[:-365, i])
        model.save_model(f"{output_dir}/xgb_{h}.json")

    # -------- LSTM --------
    if mode_flag == "FULL":
        X_seq, y_seq = make_sequences(X, y_scaled, WINDOW)
        X_t = torch.tensor(X_seq, dtype=torch.float32).to(DEVICE)
        y_t = torch.tensor(y_seq, dtype=torch.float32).to(DEVICE)

        lstm = LegendLSTM(len(features)).to(DEVICE)
        opt = torch.optim.AdamW(lstm.parameters(), lr=LR)
        loss_fn = nn.HuberLoss()

        for _ in range(EPOCHS):
            opt.zero_grad()
            loss = loss_fn(lstm(X_t), y_t)
            loss.backward()
            opt.step()

        save_file(lstm.state_dict(), f"{output_dir}/lstm.safetensors")

    # -------- META --------
    meta = {
        "ticker": ticker,
        "trained_until": str(end_date),
        "features": features,
        "window": WINDOW,
        "mode": mode_flag,
        "X_mean": X_mean.tolist(),
        "X_std": X_std.tolist(),
        "y_mean": y_mean.tolist(),
        "y_std": y_std.tolist(),
        "horizons": horizons
    }

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"✅ {ticker} DONE")
    return {"ticker": ticker, "status": "SUCCESS"}
