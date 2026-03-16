import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yfinance as yf
import xgboost as xgb
from safetensors.torch import load_file
from sklearn.preprocessing import StandardScaler

# =====================================
# CONFIG
# =====================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WINDOW = 30
HORIZONS = [90, 365]

FEATURES = [
    "Close",
    "log_ret",
    "rsi",
    "ma20",
    "ma50",
    "volatility",
    "momentum",
    "vol_chg"
]

# =====================================
# FEATURE ENGINEERING
# =====================================

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

    # 🔥 IMPORTANT FIX
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    return df

# =====================================
# LSTM MODEL
# =====================================

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

# =====================================
# PREDICTION FUNCTION
# =====================================

def predict_stock(ticker, base_dir="model_storage/model_2"):

    print(f"\n🔍 Testing prediction for {ticker}")

    model_dir = os.path.join(base_dir, ticker)

    if not os.path.exists(model_dir):
        print("❌ Model not found.")
        return

    # -------------------------
    # Load Scaler
    # -------------------------

    scaler = StandardScaler()
    scaler.mean_ = np.load(os.path.join(model_dir, "scaler_mean.npy"))
    scaler.scale_ = np.load(os.path.join(model_dir, "scaler_scale.npy"))

    # -------------------------
    # Load XGBoost
    # -------------------------

    xgb_models = []
    for h in HORIZONS:
        model = xgb.XGBRegressor()
        model.load_model(os.path.join(model_dir, f"xgb_{h}.json"))
        xgb_models.append(model)

    # -------------------------
    # Load LSTM
    # -------------------------

    lstm = StrongLSTM(len(FEATURES)).to(DEVICE)
    state_dict = load_file(os.path.join(model_dir, "lstm.safetensors"))
    lstm.load_state_dict(state_dict)
    lstm.eval()

    # -------------------------
    # Fetch latest data
    # -------------------------

    df = yf.download(ticker, period="1y", progress=False)
    df = compute_features(df)

    X = df[FEATURES].values
    if not np.isfinite(X).all():
      print("⚠ Found invalid values before scaling")
      X = np.nan_to_num(X)
    X_scaled = scaler.transform(X)

    latest_price = float(df["Close"].iloc[-1])

    # -------------------------
    # XGBoost Prediction
    # -------------------------

    xgb_pred = []
    for model in xgb_models:
        xgb_pred.append(model.predict(X_scaled[-1:].reshape(1, -1))[0])

    xgb_pred = np.array(xgb_pred)

    # -------------------------
    # LSTM Prediction
    # -------------------------

    X_seq = X_scaled[-WINDOW:]
    X_seq = torch.tensor(X_seq, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        lstm_pred = lstm(X_seq).cpu().numpy()[0]

    # -------------------------
    # Hybrid Fusion (same as training)
    # -------------------------

    final_pred = 0.6 * xgb_pred + 0.4 * lstm_pred

    # -------------------------
    # Print Results
    # -------------------------

    print(f"Current Price: ₹{latest_price:.2f}")
    pred_90_price = latest_price * (1 + final_pred[0])
    pred_365_price = latest_price * (1 + final_pred[1])

    print(f"Current Price: ₹{latest_price:.2f}")
    print(f"90-Day Return: {final_pred[0]*100:.2f}%")
    print(f"90-Day Predicted Price: ₹{pred_90_price:.2f}")
    print()
    print(f"365-Day Return: {final_pred[1]*100:.2f}%")
    print(f"365-Day Predicted Price: ₹{pred_365_price:.2f}")

    print("✅ Prediction successful")

# =====================================
# RUN TEST
# =====================================

if __name__ == "__main__":

    test_stocks = [
        "RELIANCE.NS",
        "HDFCBANK.NS",
        "BHARTIARTL.NS",
        "TCS.NS",
        "SBIN.NS"
    ]

    for stock in test_stocks:
        predict_stock(stock)
      