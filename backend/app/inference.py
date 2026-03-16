import json
import torch
import numpy as np
import pandas as pd
import yfinance as yf
import xgboost as xgb
import os
from safetensors.torch import load_file
from pathlib import Path
from app.config import MODEL_DIR


# ---------- LSTM ----------
class LegendLSTM(torch.nn.Module):
    def __init__(self, input_size, hidden=64, output=4):
        super().__init__()
        self.lstm = torch.nn.LSTM(
            input_size, hidden, batch_first=True
        )
        self.fc = torch.nn.Linear(hidden, output)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1])


# ---------- FEATURES ----------
def compute_rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = -delta.clip(upper=0).rolling(period).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))


# ---------- MAIN PREDICT ----------
def predict_ticker(ticker: str):
    model_path = MODEL_DIR / ticker
    meta_path = model_path / "meta.json"

    if not meta_path.exists():
        raise FileNotFoundError("Model not found")

    with open(meta_path) as f:
        meta = json.load(f)

    # -------- Load recent data --------
    file_path = f"data_cache/{ticker}.csv"
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} does not exist!")
    df = pd.read_csv(file_path, index_col=0, parse_dates=True)

    if df.empty:
        raise RuntimeError("No recent market data")

    df["RSI"] = compute_rsi(df["Close"])
    df["MA50"] = df["Close"].rolling(50).mean()
    df.dropna(inplace=True)

    features = meta["features"]
    X = df[features].values

    # -------- Scale --------
    X = (X - np.array(meta["X_mean"])) / np.array(meta["X_std"])

    last_close = float(df["Close"].iloc[-1])

    predictions = {}

    # -------- XGB --------
    for h in meta["horizons"]:
        xgb_model = xgb.XGBRegressor()
        xgb_model.load_model(model_path / f"xgb_{h}.json")

        r = xgb_model.predict(X[-1].reshape(1, -1))[0]
        r = r * \
            meta["y_std"][meta["horizons"].index(
                h)] + meta["y_mean"][meta["horizons"].index(h)]
        predictions[f"{h}D"] = round(last_close * (1 + r), 2)

    # -------- LSTM (optional) --------
    if meta["mode"] == "FULL":
        window = meta["window"]
        X_seq = torch.tensor(
            X[-window:].reshape(1, window, -1),
            dtype=torch.float32
        )

        lstm = LegendLSTM(len(features))
        state = load_file(model_path / "lstm.safetensors")
        lstm.load_state_dict(state)
        lstm.eval()

        with torch.no_grad():
            r = lstm(X_seq)[0].numpy()
            r = r * np.array(meta["y_std"]) + np.array(meta["y_mean"])

        for i, h in enumerate(meta["horizons"]):
            predictions[f"{h}D_LSTM"] = round(last_close * (1 + r[i]), 2)

    return {
        "ticker": ticker,
        "last_close": round(last_close, 2),
        "predictions": predictions
    }
