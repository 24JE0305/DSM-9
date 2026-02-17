import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xgboost as xgb
import yfinance as yf
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from safetensors.torch import save_file

# ==============================
# GLOBAL CONFIG
# ==============================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

WINDOW = 30
LR = 3e-4
EPOCHS = 300
PATIENCE = 20

HORIZONS = [90, 365]
MAX_HORIZON = max(HORIZONS)

# ==============================
# FEATURE ENGINEERING
# ==============================


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

# ==============================
# LSTM
# ==============================


class StrongLSTM(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, 64, num_layers=2, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(64, len(HORIZONS))

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1])
        return self.fc(out)

# ==============================
# SEQUENCE BUILDER
# ==============================


def make_sequences(X, y, window):
    Xs, ys = [], []
    for i in range(len(X) - window):
        Xs.append(X[i:i+window])
        ys.append(y[i+window])
    return np.array(Xs), np.array(ys)

# ==============================
# MAIN TRAINER
# ==============================


def train_strong_hybrid(ticker, save_dir="models"):

    os.makedirs(save_dir, exist_ok=True)

    df = yf.download(ticker, period="5y", progress=False)
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

    # ==============================
    # FEATURE SCALING (CRITICAL)
    # ==============================

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # ==============================
    # XGBOOST (Multi Model)
    # ==============================

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

    # ==============================
    # LSTM
    # ==============================

    X_seq_train, y_seq_train = make_sequences(X_train, y_train, WINDOW)
    X_seq_val, y_seq_val = make_sequences(X_val, y_val, WINDOW)

    model = StrongLSTM(len(features)).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.HuberLoss()

    best_loss = float("inf")
    patience_counter = 0

    for epoch in range(EPOCHS):

        model.train()
        optimizer.zero_grad()

        X_t = torch.tensor(X_seq_train, dtype=torch.float32).to(DEVICE)
        y_t = torch.tensor(y_seq_train, dtype=torch.float32).to(DEVICE)

        loss = loss_fn(model(X_t), y_t)
        loss.backward()
        optimizer.step()

        # Early stopping check
        model.eval()
        with torch.no_grad():
            X_val_t = torch.tensor(X_seq_val, dtype=torch.float32).to(DEVICE)
            val_pred = model(X_val_t)
            val_loss = loss_fn(
                val_pred,
                torch.tensor(y_seq_val, dtype=torch.float32).to(DEVICE)
            ).item()

        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            save_file(model.state_dict(), f"{save_dir}/lstm.safetensors")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                break

    # Reload best model
    model.load_state_dict(torch.load(f"{save_dir}/lstm.safetensors"))
    model.eval()

    with torch.no_grad():
        X_val_t = torch.tensor(X_seq_val, dtype=torch.float32).to(DEVICE)
        lstm_pred = model(X_val_t).cpu().numpy()

    # Align XGB predictions with sequence window
    xgb_pred_seq = xgb_pred[-len(lstm_pred):]

    # ==============================
    # RMSE PER HORIZON
    # ==============================

    xgb_rmse = []
    lstm_rmse = []

    for i in range(len(HORIZONS)):
        xgb_rmse.append(
            np.sqrt(mean_squared_error(y_seq_val[:, i], xgb_pred_seq[:, i]))
        )

        lstm_rmse.append(
            np.sqrt(mean_squared_error(y_seq_val[:, i], lstm_pred[:, i]))
        )

    xgb_rmse = np.array(xgb_rmse)
    lstm_rmse = np.array(lstm_rmse)

    # ==============================
    # FUSION
    # ==============================

    weight_xgb = 1 / xgb_rmse
    weight_lstm = 1 / lstm_rmse

    total = weight_xgb + weight_lstm

    weight_xgb /= total
    weight_lstm /= total

    fused_pred = (
        weight_xgb * xgb_pred_seq +
        weight_lstm * lstm_pred
    )

    fused_rmse = np.sqrt(
        mean_squared_error(y_seq_val, fused_pred)
    )

    # ==============================
    # CONFIDENCE SCORE
    # ==============================

    confidence = 1 / fused_rmse

    # Save artifacts
    for i, h in enumerate(HORIZONS):
        xgb_models[i].save_model(f"{save_dir}/xgb_{h}.json")

    np.save(f"{save_dir}/scaler_mean.npy", scaler.mean_)
    np.save(f"{save_dir}/scaler_scale.npy", scaler.scale_)

    print("FUSED RMSE:", fused_rmse)

    return {
        "fused_rmse": fused_rmse,
        "confidence_score": float(confidence),
        "weights_xgb": weight_xgb.tolist(),
        "weights_lstm": weight_lstm.tolist()
    }
