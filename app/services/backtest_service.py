# ==============================
# BACKTESTING ENGINE - MODEL 2
# app/services/backtest_service.py
# ==============================

import os
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from datetime import datetime
from pathlib import Path
from safetensors.torch import load_file
from sklearn.preprocessing import StandardScaler

from src.model_2.model_v2_0 import (
    compute_features,
    StrongLSTM,
    WINDOW,
    HORIZONS,
    DEVICE,
)

MODEL_BASE_PATH = "model_storage/model_2"

FEATURES = [
    "Close",
    "Return",
    "MA10",
    "MA50",
    "RSI",
    "Volatility",
    "Volume_Change",
    "NIFTY_Return",
]


# ==============================
# LOAD MODELS (same as inference_v2)
# ==============================

def _load_models(ticker: str):
    model_dir = os.path.join(MODEL_BASE_PATH, ticker)

    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"No trained model found for {ticker}")

    import json
    with open(os.path.join(model_dir, "metadata.json")) as f:
        metadata = json.load(f)

    scaler = StandardScaler()
    scaler.mean_ = np.load(os.path.join(model_dir, "scaler_mean.npy"))
    scaler.scale_ = np.load(os.path.join(model_dir, "scaler_scale.npy"))
    scaler.n_features_in_ = len(FEATURES)

    xgb_models = []
    for h in HORIZONS:
        m = xgb.XGBRegressor()
        m.load_model(os.path.join(model_dir, f"xgb_{h}.json"))
        xgb_models.append(m)

    lstm_model = StrongLSTM(len(FEATURES)).to(DEVICE)
    state_dict = load_file(os.path.join(model_dir, "lstm.safetensors"))
    lstm_model.load_state_dict(state_dict)
    lstm_model.eval()

    return scaler, xgb_models, lstm_model, metadata


# ==============================
# SIGNAL GENERATOR
# Predict on a slice of data → return expected 90-day return
# ==============================

def _predict_return(X_scaled: np.ndarray, xgb_models, lstm_model, metadata) -> float:
    """
    Given a scaled feature matrix, predict the 90-day expected return
    using the fused Model 2 output.
    Returns a single float (expected return).
    """
    # XGB — use last row
    xgb_preds = np.array([
        float(np.atleast_1d(m.predict(X_scaled[-1:]))[0])
        for m in xgb_models
    ])

    # LSTM — use last WINDOW rows
    if len(X_scaled) < WINDOW:
        lstm_preds = xgb_preds.copy()          # fallback: mirror XGB
    else:
        X_seq = torch.tensor(
            X_scaled[-WINDOW:].reshape(1, WINDOW, len(FEATURES)),
            dtype=torch.float32,
        ).to(DEVICE)
        with torch.no_grad():
            out = lstm_model(X_seq).cpu().numpy()
        lstm_preds = out[0] if out.ndim == 2 else out

    # Fusion weights from metadata
    w_xgb  = np.array(metadata.get("fusion_weights_xgb",  [0.5] * len(HORIZONS)))
    w_lstm = np.array(metadata.get("fusion_weights_lstm", [0.5] * len(HORIZONS)))
    fused  = w_xgb * xgb_preds + w_lstm * lstm_preds

    # Return the 90-day horizon (index 0 in HORIZONS=[90,365])
    return float(fused[0])


# ==============================
# METRICS
# ==============================

def _compute_metrics(equity_curve: np.ndarray, trade_log: list) -> dict:
    """
    equity_curve : array of portfolio values over time (starts at 1.0)
    trade_log    : list of dicts with 'signal', 'actual_return', 'correct'
    """
    total_return = float((equity_curve[-1] / equity_curve[0] - 1) * 100)

    # --- Daily returns of equity curve ---
    eq_returns = np.diff(equity_curve) / equity_curve[:-1]

    # Sharpe (annualised, assuming ~252 trading days/year)
    mean_r = eq_returns.mean()
    std_r  = eq_returns.std() + 1e-9
    sharpe = float((mean_r / std_r) * np.sqrt(252))

    # Max Drawdown
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - peak) / (peak + 1e-9)
    max_drawdown = float(drawdown.min() * 100)

    # Win Rate
    if trade_log:
        wins     = sum(1 for t in trade_log if t["correct"])
        win_rate = round(wins / len(trade_log) * 100, 2)
    else:
        win_rate = 0.0

    return {
        "total_return_pct":  round(total_return, 2),
        "sharpe_ratio":      round(sharpe, 4),
        "max_drawdown_pct":  round(max_drawdown, 2),
        "win_rate_pct":      win_rate,
        "total_trades":      len(trade_log),
    }


# ==============================
# MAIN BACKTEST FUNCTION
# ==============================

def run_backtest(
    ticker: str,
    horizon: int = 90,          # trading horizon in days
    step: int = 30,             # re-evaluate every N rows (≈ 1 month)
    min_history: int = 200,     # minimum rows before first prediction
) -> dict:
    """
    Walk-forward backtest for Model 2 on a single ticker.

    Strategy
    --------
    - Every `step` rows, predict next `horizon`-day return.
    - If predicted return > 0  →  BUY  (hold for `horizon` days).
    - If predicted return <= 0 →  SELL / stay in cash.
    - No short selling.
    - Single position sizing (all-in / all-out).

    Returns
    -------
    dict with metrics, equity curve, and trade log.
    """

    # ---- Load features & models ----
    df = compute_features(ticker)
    if df is None or df.empty:
        raise ValueError("Feature dataframe is empty")

    if len(df) < min_history + WINDOW:
        raise ValueError(
            f"Not enough data for backtest. Need {min_history + WINDOW} rows, got {len(df)}"
        )

    scaler, xgb_models, lstm_model, metadata = _load_models(ticker)

    X_raw = df[FEATURES].values
    closes = df["Close"].values
    dates  = df.index.strftime("%Y-%m-%d").tolist() if hasattr(df.index, 'strftime') else list(range(len(df)))

    # ---- Walk-forward loop ----
    equity      = 1.0
    equity_curve = [equity]
    trade_log   = []
    in_trade    = False
    trade_entry_idx   = None
    trade_entry_price = None

    i = min_history  # start after enough history

    while i < len(X_raw):

        # Scale only on data available up to row i (no lookahead)
        X_slice = X_raw[:i]
        _scaler = StandardScaler()
        X_scaled = _scaler.fit_transform(X_slice)

        # --- Close any open trade at this point ---
        if in_trade and trade_entry_idx is not None:
            exit_price   = closes[i]
            actual_return = (exit_price - trade_entry_price) / trade_entry_price
            equity       *= (1 + actual_return)
            equity_curve.append(equity)

            trade_log.append({
                "entry_date":     dates[trade_entry_idx],
                "exit_date":      dates[i],
                "entry_price":    round(float(trade_entry_price), 2),
                "exit_price":     round(float(exit_price), 2),
                "actual_return":  round(float(actual_return * 100), 2),
                "signal":         "BUY",
                "correct":        bool(actual_return > 0),
            })
            in_trade = False

        # --- Generate new signal ---
        if len(X_scaled) >= WINDOW:
            pred_return = _predict_return(X_scaled, xgb_models, lstm_model, metadata)

            if pred_return > 0:
                # Enter BUY trade
                in_trade          = True
                trade_entry_idx   = i
                trade_entry_price = closes[i]
            else:
                # Stay in cash — equity unchanged
                equity_curve.append(equity)

        i += step

    # ---- Close any remaining open trade at end ----
    if in_trade and trade_entry_idx is not None:
        exit_price    = closes[-1]
        actual_return = (exit_price - trade_entry_price) / trade_entry_price
        equity       *= (1 + actual_return)
        equity_curve.append(equity)

        trade_log.append({
            "entry_date":    dates[trade_entry_idx],
            "exit_date":     dates[-1],
            "entry_price":   round(float(trade_entry_price), 2),
            "exit_price":    round(float(exit_price), 2),
            "actual_return": round(float(actual_return * 100), 2),
            "signal":        "BUY",
            "correct":       bool(actual_return > 0),
        })

    # ---- Compute metrics ----
    equity_array = np.array(equity_curve)
    metrics = _compute_metrics(equity_array, trade_log)

    # ---- Benchmark: Buy & Hold ----
    bnh_return = float((closes[-1] / closes[min_history] - 1) * 100)

    return {
        "symbol":         ticker,
        "backtest_config": {
            "horizon_days":  horizon,
            "step_days":     step,
            "min_history":   min_history,
            "model":         "Model 2 (LSTM + XGBoost Hybrid)",
            "signal_rule":   "BUY if predicted_return > 0, else CASH",
        },
        "metrics":          metrics,
        "benchmark": {
            "strategy":           "Buy & Hold",
            "total_return_pct":   round(bnh_return, 2),
        },
        "alpha_pct":        round(metrics["total_return_pct"] - bnh_return, 2),
        "equity_curve":     [round(v, 4) for v in equity_curve],
        "trade_log":        trade_log,
        "generated_at":     datetime.utcnow().isoformat() + "Z",
        "disclaimer":       "Past performance does not guarantee future results. Not financial advice.",
    }