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

def run_backtest_v3_0_1(
    ticker: str,
    horizon: int = 90,
    step: int = 30,
    min_history: int = 200,
) -> dict:
    """Walk-forward backtest for Model 3.0 using advanced State Machine logic."""

    df = compute_features_v3(ticker)
    if df is None or df.empty:
        raise ValueError("Feature dataframe is empty")
    
    # WINDOW is imported from model_v3_0.py, which is now 120
    if len(df) < min_history + WINDOW:
        raise ValueError(f"Need {min_history + WINDOW} rows, got {len(df)}")

    scaler, xgb_models, deep_model, metadata = _load_v3(ticker)

    X_raw  = df[FEATURES_V3].values
    closes = df["Close"].values
    dates  = df.index.strftime("%Y-%m-%d").tolist() if hasattr(df.index, "strftime") else list(range(len(df)))

    # ── HYPERPARAMETERS FOR EXCELLENT LOGIC ──────────────────────
    ENTRY_MIN_PRED   = 0.03  # AI must predict at least a 3% gain over 90 days to BUY
    EXIT_BEARISH     = 0.00  # AI predicts the stock will drop or stay flat
    STOP_LOSS_PCT    = 0.08  # HARD EXIT: Sell immediately if down 8%
    TAKE_PROFIT_PCT  = 0.30  # HARD EXIT: Lock in gains if up 30%
    # ─────────────────────────────────────────────────────────────

    equity       = 1.0
    equity_curve = [equity]
    trades       = []
    
    in_trade     = False
    entry_idx    = None
    entry_price  = None

    i = min_history
    while i < len(X_raw):
        current_price = closes[i]
        current_date  = dates[i]
        
        # FIX: Use the loaded scaler's transform, NOT fit_transform!
        X_scaled = scaler.transform(X_raw[:i])

        # Get AI Prediction for the 90-day horizon
        pred = 0.0
        if len(X_scaled) >= WINDOW:
            pred = _predict_return_v3(X_scaled, xgb_models, deep_model)

        # ── STATE: WE CURRENTLY OWN THE STOCK ────────────────────────
        if in_trade:
            current_return = (current_price - entry_price) / entry_price
            days_held = i - entry_idx # Approximate days held
            
            # Check Exit Rules
            hit_stop_loss   = current_return <= -STOP_LOSS_PCT
            hit_take_profit = current_return >= TAKE_PROFIT_PCT
            time_to_sell    = (days_held >= horizon) and (pred <= EXIT_BEARISH)
            
            if hit_stop_loss or hit_take_profit or time_to_sell:
                equity *= (1 + current_return)
                
                if hit_stop_loss: reason = "STOP_LOSS"
                elif hit_take_profit: reason = "TAKE_PROFIT"
                else: reason = "AI_BEARISH_FLIP"

                trades.append({
                    "entry_date":    dates[entry_idx],
                    "exit_date":     current_date,
                    "entry_price":   round(float(entry_price), 2),
                    "exit_price":    round(float(current_price), 2),
                    "actual_return": round(float(current_return * 100), 2),
                    "signal":        f"SELL ({reason})",
                    "correct":       bool(current_return > 0),
                })
                in_trade    = False
                entry_idx   = None
                entry_price = None

        # ── STATE: WE ARE SITTING IN CASH ────────────────────────────
        if not in_trade:
            # AI must have strong conviction on the 90-day horizon
            if pred >= ENTRY_MIN_PRED:
                in_trade    = True
                entry_idx   = i
                entry_price = current_price

        # ── RECORD MARK-TO-MARKET METRICS ────────────────────────────
        if in_trade:
            paper_return = (current_price - entry_price) / entry_price
            current_equity = equity * (1 + paper_return)
        else:
            current_equity = equity
            
        # Avoid duplicating the initial equity value on the very first step
        if i > min_history:
            equity_curve.append(current_equity)

        i += step

    # ── CLOSE FINAL TRADE (End of Data) ───────────────────────────
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
            "signal":        "SELL (END_OF_DATA)",
            "correct":       bool(ret > 0),
        })

    eq_arr  = np.array(equity_curve)
    metrics = _metrics(eq_arr, trades)
    bnh     = float((closes[-1] / closes[min_history] - 1) * 100)

    return {
        "symbol": ticker,
        "backtest_config": {
            "model":        "Model 3.1 (State Machine Focus)",
            "horizon_days": horizon,
            "step_days":    step,
            "min_history":  min_history,
            "signal_rule":  f"BUY if > {ENTRY_MIN_PRED*100}%, SELL on TP/SL or Bearish Flip",
        },
        "metrics":      metrics,
        "benchmark":    {"strategy": "Buy & Hold", "total_return_pct": round(bnh, 2)},
        "alpha_pct":    round(metrics["total_return_pct"] - bnh, 2),
        "equity_curve": [round(v, 4) for v in equity_curve],
        "trade_log":    trades,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "disclaimer":   "Past performance does not guarantee future results. Not financial advice.",
    }