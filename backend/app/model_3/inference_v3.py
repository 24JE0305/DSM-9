# ============================================================
# DSM-9  MODEL 3.0 — INFERENCE
# app/model_3/inference_v3.py
#
# Intentionally mirrors inference_v2.py API so all downstream
# endpoints (backtest, report) work identically with v3.
# ============================================================

import numpy as np
import torch
from datetime import datetime

from src.model_3.features_v3 import compute_features_v3, FEATURES_V3
from src.model_3.model_v3_0 import WINDOW, HORIZONS
from app.model_3.mongo_loader import load_v3_models_from_mongo


# ── Helpers (identical contract to v2) ───────────────────────

def confidence_label(score: float) -> str:
    if score >= 0.8: return "High"
    if score >= 0.6: return "Moderate"
    return "Low"


def signal_bias(ret: float) -> str:
    if ret >  0.15: return "Strong Bullish"
    if ret >  0:    return "Bullish"
    if ret < -0.15: return "Strong Bearish"
    if ret <  0:    return "Bearish"
    return "Neutral"


# ── Model loader (MongoDB-backed) ────────────────────────────
# load_v3_models_from_mongo() returns the same tuple as the old
# _load_v3_models() — (scaler, xgb_models, deep_model, metadata)


# ── Main inference function ───────────────────────────────────

def predict_v3(ticker: str) -> dict:
    """
    Run Model 3.0 inference for a ticker.
    Returns the same JSON structure as predict_v2() so all
    downstream services (report, backtest) work without changes.
    """

    # ── Features ────────────────────────────────────────────
    df = compute_features_v3(ticker)

    if df is None or df.empty:
        raise ValueError("Feature dataframe is empty")
    if len(df) < WINDOW:
        raise ValueError(f"Need at least {WINDOW} rows, got {len(df)}")

    last_close = float(df["Close"].iloc[-1])
    # ✅ After — use ATR/Close ratio instead (proper v3 volatility measure)
    volatility = float(df["ATR"].iloc[-1] / df["Close"].iloc[-1]) if "ATR" in df.columns else 0.02

    X = df[FEATURES_V3].values

    # ── Load models from MongoDB ─────────────────────────────
    scaler, xgb_models, deep_model, metadata = load_v3_models_from_mongo(ticker)

    # ── Scale ───────────────────────────────────────────────
    X_scaled = scaler.transform(X)

    # ── XGBoost predictions ─────────────────────────────────
    xgb_preds = np.array([
        float(np.atleast_1d(m.predict(X_scaled[-1:]))[0])
        for m in xgb_models
    ])

    # ── Deep model prediction ────────────────────────────────
    X_seq = torch.tensor(
        X_scaled[-WINDOW:].reshape(1, WINDOW, len(FEATURES_V3)),
        dtype=torch.float32,
    ).to(DEVICE)

    xgb_tensor = torch.tensor(
        xgb_preds.reshape(1, -1), dtype=torch.float32
    ).to(DEVICE)

    with torch.no_grad():
        deep_out = deep_model(X_seq, xgb_tensor).cpu().numpy()[0]

    # ── Model agreement ─────────────────────────────────────
    agreement = float(
        max(0.0, min(1.0, 1 - np.mean(np.abs(xgb_preds - deep_out))))
    )

    # ── Build predictions dict (same shape as v2) ───────────
    fused_rmse_raw = metadata.get("val_rmse", 0.02)
    fused_rmse     = np.array([fused_rmse_raw] * len(HORIZONS))
    conf_score     = float(metadata.get("confidence_score", 0.6))

    predictions = {}
    for i, h in enumerate(HORIZONS):
        er          = float(deep_out[i])
        low         = er - fused_rmse[i]
        high        = er + fused_rmse[i]
        predictions[f"{h}_days"] = {
            "expected_return":    round(er, 4),
            "return_percentage":  round(er * 100, 2),
            "confidence_score":   round(conf_score, 2),
            "confidence_level":   confidence_label(conf_score),
            "range": {
                "low":      round(low, 4),
                "expected": round(er,  4),
                "high":     round(high, 4),
            },
            "signal_bias": signal_bias(er),
        }

    return {
        "symbol":       ticker,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "model_info": {
            "version":          metadata.get("version", "3.0"),
            "type":             "DSM9-v3 (Transformer + BiLSTM-Attention + XGBoost + LearnedFusion)",
            "training_window":  "5 Years",
            "horizons":         HORIZONS,
            "last_retrained":   metadata.get("last_retrained", "unknown"),
            "n_features":       len(FEATURES_V3),
            "architecture":     metadata.get("architecture", ""),
        },
        "predictions":  predictions,
        "risk_metrics": {
            "volatility_level": confidence_label(1 - volatility),
            "model_agreement":  round(agreement, 2),
        },
        "disclaimer": "AI-generated probabilistic forecast. Not financial advice.",
    }
