import os
import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
import torch.nn as nn
import xgboost as xgb
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

# ==============================
# FEATURE ENGINEERING
# ==============================

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def compute_features(ticker):

    cache_dir = Path("data_cache")

    ticker_path = cache_dir / f"{ticker}.csv"
    nifty_path = cache_dir / "^NSEI.csv"

    if not ticker_path.exists():
        raise FileNotFoundError(f"Ticker file not found: {ticker_path}")

    if not nifty_path.exists():
        raise FileNotFoundError(f"NIFTY file not found: {nifty_path}")

    df = pd.read_csv(ticker_path, index_col=0, parse_dates=True)
    nifty = pd.read_csv(nifty_path, index_col=0, parse_dates=True)

    print("Loaded ticker shape:", df.shape)
    print("Loaded nifty shape:", nifty.shape)

    required_cols = ["Close", "Volume"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' in ticker data")

    if "Close" not in nifty.columns:
        raise ValueError("Missing 'Close' in NIFTY data")

    nifty["NIFTY_Return"] = nifty["Close"].pct_change()
    df = df.join(nifty["NIFTY_Return"], how="left")

    df["Return"] = df["Close"].pct_change()
    df["MA10"] = df["Close"].rolling(10).mean()
    df["MA50"] = df["Close"].rolling(50).mean()
    df["RSI"] = compute_rsi(df["Close"])
    df["Volatility"] = df["Return"].rolling(10).std()
    df["Volume_Change"] = df["Volume"].pct_change()

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.ffill(inplace=True)
    df.dropna(inplace=True)

    print("After feature engineering shape:", df.shape)

    if df.isna().sum().sum() > 0:
        raise ValueError("NaNs still exist after cleaning")

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
    Xs = np.array(Xs)
    ys = np.array(ys)

    print("Sequence X shape:", Xs.shape)
    print("Sequence y shape:", ys.shape)

    return Xs, ys


# ==============================
# MAIN TRAINER
# ==============================

def train_strong_hybrid(ticker, save_dir="model_storage/model_2"):

    os.makedirs(save_dir, exist_ok=True)

    df = compute_features(ticker)

    if df is None or len(df) < 400:
        raise ValueError("Not enough clean data")

    for h in HORIZONS:
        df[f"target_{h}"] = df["Close"].shift(-h) / df["Close"] - 1

    df.dropna(inplace=True)

    features = [
        "Close",
        "Return",
        "MA10",
        "MA50",
        "RSI",
        "Volatility",
        "Volume_Change",
        "NIFTY_Return"
    ]

    for col in features:
        if col not in df.columns:
            raise ValueError(f"Missing feature column: {col}")

    X = df[features].values
    y = df[[f"target_{h}" for h in HORIZONS]].values

    print("X shape:", X.shape)
    print("y shape:", y.shape)

    if X.ndim != 2:
        raise ValueError("X is not 2D")

    if y.ndim != 2:
        raise ValueError("y is not 2D")

    train_split = int(len(X) * 0.7)
    val_split = int(len(X) * 0.85)

    X_train = X[:train_split]
    y_train = y[:train_split]
    X_val = X[train_split:val_split]
    y_val = y[train_split:val_split]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # ==============================
    # XGBOOST
    # ==============================

    xgb_models = []
    xgb_preds = []

    for i in range(len(HORIZONS)):

        if y_train.ndim != 2:
            raise ValueError("y_train collapsed to 1D!")

        model_xgb = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.02,
            tree_method="hist",
            random_state=SEED
        )

        model_xgb.fit(X_train, y_train[:, i])
        preds = model_xgb.predict(X_val)

        print(f"XGB preds shape (h={HORIZONS[i]}):", preds.shape)

        xgb_models.append(model_xgb)
        xgb_preds.append(preds)

    xgb_pred = np.column_stack(xgb_preds)
    print("Stacked XGB shape:", xgb_pred.shape)

    # ==============================
    # LSTM
    # ==============================

    X_seq_train, y_seq_train = make_sequences(X_train, y_train, WINDOW)
    X_seq_val, y_seq_val = make_sequences(X_val, y_val, WINDOW)

    if len(X_seq_train) == 0:
        raise ValueError("Not enough data for LSTM window.")

    model = StrongLSTM(len(features)).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.HuberLoss()

    best_loss = float("inf")
    best_weights = model.state_dict().copy()
    patience_counter = 0

    for epoch in range(EPOCHS):

        model.train()
        optimizer.zero_grad()

        X_t = torch.tensor(X_seq_train, dtype=torch.float32).to(DEVICE)
        y_t = torch.tensor(y_seq_train, dtype=torch.float32).to(DEVICE)

        loss = loss_fn(model(X_t), y_t)
        loss.backward()
        optimizer.step()

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
            best_weights = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                break

    model.load_state_dict(best_weights)

    print("LSTM training completed.")

    print("Final validation shapes:")
    print("y_seq_val:", y_seq_val.shape)
    print("xgb_pred:", xgb_pred.shape)

    return {"status": "Training completed successfully"}