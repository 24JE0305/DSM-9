# src/trainer_v2_0.py

import os
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
import yfinance as yf

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from torch.optim import Adam
from torch.nn import HuberLoss

from src.model_v2_0 import StrongLSTM_v2


# =========================
# GLOBAL CONFIG
# =========================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WINDOW = 30
HORIZONS = [90, 365]
EPOCHS = 300
PATIENCE = 20
LR = 3e-4
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)


# =========================
# FEATURE ENGINEERING
# =========================

def compute_rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = -delta.clip(upper=0).rolling(period).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))


def compute_features(df):
    df["log_ret"] = np.log(df["Close"] / df["Close"].shift(1))
    df["rsi"] = compute_rsi(df["Close"])
    df["ma20"] = df["Close"].rolling(20).mean()
    df["ma50"] = df["Close"].rolling(50).mean()
    df["volatility"] = df["log_ret"].rolling(20).std()
    df["momentum"] = df["Close"].pct_change(10)
    df["vol_chg"] = df["Volume"].pct_change()
    df.dropna(inplace=True)
    return df


def make_sequences(X, y, window):
    Xs, ys = [], []
    for i in range(len(X) - window):
        Xs.append(X[i:i+window])
        ys.append(y[i+window])
    return np.array(Xs), np.array(ys)


# =========================
# MAIN TRAINER
# =========================

def train_v2(ticker):

    print(f"\nTraining v2.0 model for {ticker}")

    df = yf.download(ticker, period="5y", progress=False)

    if df.empty:
        raise ValueError("No data returned.")

    df = compute_features(df)

    for h in HORIZONS:
        df[f"target_{h}"] = df["Close"].shift(-h) / df["Close"] - 1

    df.dropna(inplace=True)

    features = [
        "Close", "log_ret", "rsi", "ma20", "ma50",
        "volatility", "momentum", "vol_chg"
    ]

    X = df[features].values
    y = df[[f"target_{h}" for h in HORIZONS]].values

    split = int(len(X) * 0.8)

    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    # =========================
    # SCALING
    # =========================

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # =========================
    # XGBOOST
    # =========================

    xgb_models = []
    xgb_preds = []

    for i, h in enumerate(HORIZONS):

        model = xgb.XGBRegressor(
            n_estimators=800,
            max_depth=5,
            learning_rate=0.02,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            random_state=SEED
        )

        model.fit(
            X_train,
            y_train[:, i],
            eval_set=[(X_val, y_val[:, i])],
            verbose=False
        )

        xgb_models.append(model)
        xgb_preds.append(model.predict(X_val))

    xgb_pred = np.column_stack(xgb_preds)

    # =========================
    # LSTM
    # =========================

    X_seq_train, y_seq_train = make_sequences(X_train, y_train, WINDOW)
    X_seq_val, y_seq_val = make_sequences(X_val, y_val, WINDOW)

    model = StrongLSTM_v2(
        input_size=len(features),
        output_size=len(HORIZONS)
    ).to(DEVICE)

    optimizer = Adam(model.parameters(), lr=LR)
    loss_fn = HuberLoss()

    best_loss = float("inf")
    patience_counter = 0

    for epoch in range(EPOCHS):

        model.train()

        X_t = torch.tensor(X_seq_train, dtype=torch.float32).to(DEVICE)
        y_t = torch.tensor(y_seq_train, dtype=torch.float32).to(DEVICE)

        optimizer.zero_grad()
        loss = loss_fn(model(X_t), y_t)
        loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            X_val_t = torch.tensor(X_seq_val, dtype=torch.float32).to(DEVICE)
            val_pred = model(X_val_t)
            val_loss = loss_fn(val_pred,
                               torch.tensor(y_seq_val, dtype=torch.float32).to(DEVICE))

        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch}")
            break

    model.eval()
    with torch.no_grad():
        lstm_pred = model(
            torch.tensor(X_seq_val, dtype=torch.float32).to(DEVICE)
        ).cpu().numpy()

    # =========================
    # METRICS
    # =========================

    xgb_rmse = np.sqrt(mean_squared_error(y_val, xgb_pred))
    lstm_rmse = np.sqrt(mean_squared_error(y_seq_val, lstm_pred))

    print("XGB RMSE:", xgb_rmse)
    print("LSTM RMSE:", lstm_rmse)

    return {
        "xgb_rmse": xgb_rmse,
        "lstm_rmse": lstm_rmse
    }
