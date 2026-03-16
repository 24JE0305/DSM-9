# ============================================================
# DSM-9  MODEL 3.0 — BACKTEST SERVICE
# app/services/backtest_v3.py
#
# Mirrors backtest_service.py (v2) exactly — only the
# model loading and inference calls are swapped to v3.
# ============================================================

import os
import json
import numpy as np
import torch
import xgboost as xgb
from datetime import datetime
from pathlib import Path
from safetensors.torch import load_file
from sklearn.preprocessing import StandardScaler

from src.model_3.features_v3 import compute_features_v3, FEATURES_V3
from src.model_3.model_v3_0 import DSM9_v3, DEVICE, WINDOW, HORIZONS

MODEL_BASE = Path("model_storage/model_3")


# ── Model loader ─────────────────────────────────────────────

def _load_v3(ticker: str):
    d = MODEL_BASE / ticker
    if not d.is_dir():
        raise FileNotFoundError(f"No Model 3.0 for {ticker}")

    with open(d / "metadata.json") as f:
        meta = json.load(f)

    sc = StandardScaler()
    sc.mean_          = np.load(d / "scaler_mean.npy")
    sc.scale_         = np.load(d / "scaler_scale.npy")
    sc.n_features_in_ = len(FEATURES_V3)

    xgbs = []
    for h in HORIZONS:
        m = xgb.XGBRegressor()
        m.load_model(str(d / f"xgb_{h}.json"))
        xgbs.append(m)

    model = DSM9_v3(n_feat=len(FEATURES_V3)).to(DEVICE)
    model.load_state_dict(load_file(str(d / "model_v3.safetensors")))
    model.eval()

    return sc, xgbs, model, meta


# ── Single-step predictor ─────────────────────────────────────

def _predict_return_v3(X_scaled, xgb_models, deep_model) -> float:
    """Predict 90-day return from a scaled feature slice."""
    xgb_preds = np.array([
        float(np.atleast_1d(m.predict(X_scaled[-1:]))[0])
        for m in xgb_models
    ])

    if len(X_scaled) < WINDOW:
        return float(xgb_preds[0])   # fallback: XGB only

    X_seq = torch.tensor(
        X_scaled[-WINDOW:].reshape(1, WINDOW, len(FEATURES_V3)),
        dtype=torch.float32,
    ).to(DEVICE)
    xgb_t = torch.tensor(xgb_preds.reshape(1, -1), dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        out = deep_model(X_seq, xgb_t).cpu().numpy()[0]

    return float(out[0])   # 90-day horizon


# ── Metrics ──────────────────────────────────────────────────

def _metrics(equity: np.ndarray, trades: list) -> dict:
    total_ret = float((equity[-1] / equity[0] - 1) * 100)
    eq_ret    = np.diff(equity) / equity[:-1]
    sharpe    = float((eq_ret.mean() / (eq_ret.std() + 1e-9)) * np.sqrt(252))
    peak      = np.maximum.accumulate(equity)
    max_dd    = float(((equity - peak) / (peak + 1e-9)).min() * 100)
    win_rate  = round(sum(1 for t in trades if t["correct"]) / len(trades) * 100, 2) if trades else 0.0
    return {
        "total_return_pct": round(total_ret, 2),
        "sharpe_ratio":     round(sharpe, 4),
        "max_drawdown_pct": round(max_dd, 2),
        "win_rate_pct":     win_rate,
        "total_trades":     len(trades),
    }


# ── Main backtest ─────────────────────────────────────────────

def run_backtest_v3(
    ticker: str,
    horizon: int = 90,
    step: int = 30,
    min_history: int = 200,
) -> dict:
    """Walk-forward backtest for Model 3.0 — identical contract to run_backtest()."""

    df = compute_features_v3(ticker)
    if df is None or df.empty:
        raise ValueError("Feature dataframe is empty")
    if len(df) < min_history + WINDOW:
        raise ValueError(f"Need {min_history + WINDOW} rows, got {len(df)}")

    scaler, xgb_models, deep_model, metadata = _load_v3(ticker)

    X_raw  = df[FEATURES_V3].values
    closes = df["Close"].values
    dates  = df.index.strftime("%Y-%m-%d").tolist() if hasattr(df.index, "strftime") else list(range(len(df)))

    equity       = 1.0
    equity_curve = [equity]
    trades       = []
    in_trade     = False
    entry_idx    = None
    entry_price  = None

    i = min_history
    while i < len(X_raw):
        X_scaled = StandardScaler().fit_transform(X_raw[:i])

        # Close open trade
        if in_trade and entry_idx is not None:
            ret    = (closes[i] - entry_price) / entry_price
            equity *= (1 + ret)
            equity_curve.append(equity)
            trades.append({
                "entry_date":    dates[entry_idx],
                "exit_date":     dates[i],
                "entry_price":   round(float(entry_price), 2),
                "exit_price":    round(float(closes[i]), 2),
                "actual_return": round(float(ret * 100), 2),
                "signal":        "BUY",
                "correct":       bool(ret > 0),
            })
            in_trade = False

        # New signal
        if len(X_scaled) >= WINDOW:
            pred = _predict_return_v3(X_scaled, xgb_models, deep_model)
            if pred > 0:
                in_trade    = True
                entry_idx   = i
                entry_price = closes[i]
            else:
                equity_curve.append(equity)

        i += step

    # Close final trade
    if in_trade and entry_idx is not None:
        ret    = (closes[-1] - entry_price) / entry_price
        equity *= (1 + ret)
        equity_curve.append(equity)
        trades.append({
            "entry_date":    dates[entry_idx],
            "exit_date":     dates[-1],
            "entry_price":   round(float(entry_price), 2),
            "exit_price":    round(float(closes[-1]), 2),
            "actual_return": round(float(ret * 100), 2),
            "signal":        "BUY",
            "correct":       bool(ret > 0),
        })

    eq_arr  = np.array(equity_curve)
    metrics = _metrics(eq_arr, trades)
    bnh     = float((closes[-1] / closes[min_history] - 1) * 100)

    return {
        "symbol": ticker,
        "backtest_config": {
            "model":        "Model 3.0 (Transformer + BiLSTM-Attention + XGBoost)",
            "horizon_days": horizon,
            "step_days":    step,
            "min_history":  min_history,
            "signal_rule":  "BUY if predicted_return > 0, else CASH",
        },
        "metrics":      metrics,
        "benchmark":    {"strategy": "Buy & Hold", "total_return_pct": round(bnh, 2)},
        "alpha_pct":    round(metrics["total_return_pct"] - bnh, 2),
        "equity_curve": [round(v, 4) for v in equity_curve],
        "trade_log":    trades,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "disclaimer":   "Past performance does not guarantee future results. Not financial advice.",
    }
