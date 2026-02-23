import os
import json
import numpy as np
import torch
import xgboost as xgb
from datetime import datetime
from safetensors.torch import load_file
from sklearn.preprocessing import StandardScaler

from src.model_2.model_v2_0 import (
    compute_features,
    StrongLSTM,
    WINDOW,
    HORIZONS,
    DEVICE
)

MODEL_BASE_PATH = "model_storage/model_2"
DATA_CACHE = "data_cache_model_2"

FEATURES = [
    "Close",
    "Return",
    "MA10",
    "MA50",
    "RSI",
    "Volatility",
    "Volume_Change",
    "NIFTY_Return"
]


# ==============================
# SAFE MODEL LOADER
# ==============================

def load_models(ticker):

    model_dir = os.path.join(MODEL_BASE_PATH, ticker)

    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"No trained model found for {ticker}")

    # ---------- Load Metadata ----------
    metadata_path = os.path.join(model_dir, "metadata.json")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError("metadata.json missing")

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    # ---------- Load Scaler ----------
    mean_path = os.path.join(model_dir, "scaler_mean.npy")
    scale_path = os.path.join(model_dir, "scaler_scale.npy")

    if not os.path.exists(mean_path) or not os.path.exists(scale_path):
        raise FileNotFoundError("Scaler parameters missing")

    scaler = StandardScaler()
    scaler.mean_ = np.load(mean_path)
    scaler.scale_ = np.load(scale_path)
    scaler.n_features_in_ = len(FEATURES)

    # ---------- Load XGBoost ----------
    xgb_models = []
    for h in HORIZONS:
        model_path = os.path.join(model_dir, f"xgb_{h}.json")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Missing XGB model for horizon {h}")

        model = xgb.XGBRegressor()
        model.load_model(model_path)
        xgb_models.append(model)

    # ---------- Load LSTM ----------
    lstm_path = os.path.join(model_dir, "lstm.safetensors")
    if not os.path.exists(lstm_path):
        raise FileNotFoundError("Missing LSTM model")

    lstm_model = StrongLSTM(len(FEATURES)).to(DEVICE)
    state_dict = load_file(lstm_path)
    lstm_model.load_state_dict(state_dict)
    lstm_model.eval()

    return scaler, xgb_models, lstm_model, metadata


# ==============================
# HELPERS
# ==============================

def confidence_label(score):
    if score >= 0.8:
        return "High"
    elif score >= 0.6:
        return "Moderate"
    return "Low"


def signal_bias(ret):
    if ret > 0.15:
        return "Strong Bullish"
    if ret > 0:
        return "Bullish"
    if ret < -0.15:
        return "Strong Bearish"
    if ret < 0:
        return "Bearish"
    return "Neutral"


# ==============================
# SAFE INFERENCE
# ==============================

def predict_v2(ticker: str):

    # ---------- Compute Features ----------
    df = compute_features(ticker)

    if df is None or df.empty:
        raise ValueError("Feature dataframe is empty")

    if len(df) < WINDOW:
        raise ValueError(f"Not enough data. Required: {WINDOW}, Got: {len(df)}")
    print("done126")
    # Check required columns
    missing_cols = [col for col in FEATURES if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required feature columns: {missing_cols}")
    print("done131")
    last_close = float(df["Close"].iloc[-1])
    volatility = float(df["Volatility"].iloc[-1])

    X = df[FEATURES].values

    if X.ndim != 2:
        raise ValueError("Feature matrix is not 2D")

    # ---------- Load Models ----------
    scaler, xgb_models, lstm_model, metadata = load_models(ticker)

    # ---------- Scale ----------
    X_scaled = scaler.transform(X)

    # ---------- XGB ----------
    xgb_preds = []

    for model in xgb_models:
        print("done150")
        last_row = X_scaled[-1:].copy()   # Always 2D
        preds = model.predict(last_row)

        preds = np.atleast_1d(preds)      # Force array
        print("done155")
        xgb_preds.append(float(preds[0]))

    xgb_preds = np.array(xgb_preds)
    print("done159")
    # ---------- LSTM ----------
    X_seq = X_scaled[-WINDOW:]
    X_seq = torch.tensor(
        X_seq.reshape(1, WINDOW, len(FEATURES)),
        dtype=torch.float32
    ).to(DEVICE)

    with torch.no_grad():
        lstm_out = lstm_model(X_seq).cpu().numpy()

    if lstm_out.ndim == 1:
        lstm_preds = lstm_out
    else:
        lstm_preds = lstm_out[0]
    print("done174")
    # ---------- Fusion Safety ----------
    w_xgb = np.array(metadata.get("fusion_weights_xgb", [0.5]*len(HORIZONS)))
    w_lstm = np.array(metadata.get("fusion_weights_lstm", [0.5]*len(HORIZONS)))
    fused_rmse_raw = metadata.get("fused_rmse", 0.02)

    # If scalar → replicate for all horizons
    if np.isscalar(fused_rmse_raw):
        fused_rmse = np.array([fused_rmse_raw] * len(HORIZONS))
    else:
        fused_rmse = np.array(fused_rmse_raw)

    # Final safety
    if fused_rmse.ndim == 0:
        fused_rmse = np.array([float(fused_rmse)] * len(HORIZONS))

    if not (len(w_xgb) == len(w_lstm) == len(HORIZONS)):
        raise ValueError("Fusion weight size mismatch")

    fused = w_xgb * xgb_preds + w_lstm * lstm_preds

    # ---------- Model Agreement ----------
    agreement = float(
        1 - np.mean(np.abs(xgb_preds - lstm_preds))
    )
    agreement = max(0.0, min(1.0, agreement))  # clamp

    predictions = {}
    print("done192")
    for i, h in enumerate(HORIZONS):
        print("done194")
        expected_return = float(fused[i])
        return_pct = round(expected_return * 100, 2)
        print("done197")
        conf_score = float(metadata.get("confidence_score", 0.6))
        conf_level = confidence_label(conf_score)
        print("done200")
        print("fused_rmse:", fused_rmse)
        print("shape:", np.shape(fused_rmse))
        low = expected_return - fused_rmse[i]
        high = expected_return + fused_rmse[i]
        print("done203")
        predictions[f"{h}_days"] = {
            "expected_return": round(expected_return, 4),
            "return_percentage": return_pct,
            "confidence_score": round(conf_score, 2),
            "confidence_level": conf_level,
            "range": {
                "low": round(low, 4),
                "expected": round(expected_return, 4),
                "high": round(high, 4)
            },
            "signal_bias": signal_bias(expected_return)
        }
    print("done216")
    return {
        "symbol": ticker,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "model_info": {
            "version": metadata.get("version", "unknown"),
            "type": "Hybrid (LSTM + XGBoost)",
            "training_window": "5 Years",
            "horizons": HORIZONS,
            "last_retrained": metadata.get("last_retrained", "unknown")
        },
        "predictions": predictions,
        "risk_metrics": {
            "volatility_level": confidence_label(1 - volatility),
            "model_agreement": round(agreement, 2)
        },
        "disclaimer": "This is an AI-generated probabilistic forecast and not financial advice."
    }