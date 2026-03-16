# ============================================================
# DSM-9  MODEL 3.0 — TRUE WALK-FORWARD ACCURACY BACKTEST
# run_accuracy_test.py
#
# Run from project root:
#   python run_accuracy_test.py
#
# Uses the EXACT same 24-feature set and compute logic as the
# training notebook (Cell 5) — no more missing feature errors.
#
# True walk-forward: features are recomputed on a historical
# slice per window, so each window gives a genuinely different
# prediction (no frozen values).
# ============================================================

import sys
import json
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from datetime import datetime
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from safetensors.torch import load_file

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent))

from src.model_3.model_v3_0 import DSM9_v3, DEVICE, WINDOW, HORIZONS

DATA_CACHE = Path("data_cache")
MODEL_BASE = Path("model_storage/model_3")

# ── Exact feature list from training notebook (Cell 3) ───────
FEATURES_V3 = [
    "Close", "Return", "Log_Return",
    "MA10", "MA50", "MA200", "EMA20", "Price_vs_MA50", "Price_vs_MA200",
    "RSI", "MACD", "MACD_Signal", "MACD_Hist",
    "BB_Upper", "BB_Lower", "BB_Width", "ATR",
    "OBV", "Volume_Change", "Volume_MA20",
    "NIFTY_Return", "NIFTY_MA10",
    "FII_Net", "DII_Net",
]

with open("data/nifty_top50.json") as f:
    TICKERS = json.load(f)["tickers"]


# ── Exact helper functions from training notebook (Cell 5) ───

def compute_rsi(s, p=14):
    d = s.diff()
    g = d.where(d > 0, 0).rolling(p).mean()
    l = (-d.where(d < 0, 0)).rolling(p).mean()
    return 100 - 100 / (1 + g / (l + 1e-9))

def compute_macd(s, fast=12, slow=26, sig=9):
    ef = s.ewm(span=fast, adjust=False).mean()
    es = s.ewm(span=slow, adjust=False).mean()
    m  = ef - es
    sl = m.ewm(span=sig, adjust=False).mean()
    return m, sl, m - sl

def compute_bb(s, p=20, n=2):
    ma  = s.rolling(p).mean()
    std = s.rolling(p).std()
    u   = ma + n * std
    l   = ma - n * std
    return u, l, (u - l) / (ma + 1e-9)

def compute_atr(h, l, c, p=14):
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(p).mean()

def compute_obv(c, v):
    return (np.sign(c.diff().fillna(0)) * v).cumsum()

def load_fii_dii():
    p = DATA_CACHE / "FII_DII.csv"
    if not p.exists():
        return pd.DataFrame(columns=["FII_Net", "DII_Net"])
    df = pd.read_csv(p, index_col=0, parse_dates=True)[["FII_Net", "DII_Net"]]
    return df.apply(pd.to_numeric, errors="coerce").fillna(0)


# ── Feature computation on a sliced DataFrame ────────────────
def compute_features_from_slice(df_slice: pd.DataFrame,
                                 nifty_full: pd.DataFrame,
                                 fii_dii: pd.DataFrame) -> pd.DataFrame:
    """
    Recompute all 24 features from a pre-sliced ticker DataFrame.
    Uses the exact same logic as compute_features() in the notebook.
    nifty_full and fii_dii are already loaded — we join by index.
    """
    df = df_slice.copy()

    df["Return"]     = df["Close"].pct_change()
    df["Log_Return"] = np.log(df["Close"] / df["Close"].shift(1))
    df["MA10"]       = df["Close"].rolling(10).mean()
    df["MA50"]       = df["Close"].rolling(50).mean()
    df["MA200"]      = df["Close"].rolling(200).mean()
    df["EMA20"]      = df["Close"].ewm(span=20, adjust=False).mean()
    df["Price_vs_MA50"]  = (df["Close"] - df["MA50"])  / (df["MA50"]  + 1e-9)
    df["Price_vs_MA200"] = (df["Close"] - df["MA200"]) / (df["MA200"] + 1e-9)
    df["RSI"]        = compute_rsi(df["Close"])
    df["MACD"], df["MACD_Signal"], df["MACD_Hist"] = compute_macd(df["Close"])
    df["BB_Upper"], df["BB_Lower"], df["BB_Width"]  = compute_bb(df["Close"])
    df["ATR"]           = compute_atr(df["High"], df["Low"], df["Close"])
    df["OBV"]           = compute_obv(df["Close"], df["Volume"])
    df["Volume_Change"] = df["Volume"].pct_change()
    df["Volume_MA20"]   = df["Volume"].rolling(20).mean()

    # Join NIFTY columns (slice nifty to same date range)
    nifty = nifty_full.copy()
    nifty["NIFTY_Return"] = nifty["Close"].pct_change()
    nifty["NIFTY_MA10"]   = nifty["Close"].rolling(10).mean()
    df = df.join(nifty[["NIFTY_Return", "NIFTY_MA10"]], how="left")

    # Join FII/DII
    if not fii_dii.empty:
        df = df.join(fii_dii, how="left")
    else:
        df["FII_Net"] = 0.0
        df["DII_Net"] = 0.0

    df["FII_Net"] = df.get("FII_Net", pd.Series(0.0, index=df.index)).fillna(0)
    df["DII_Net"] = df.get("DII_Net", pd.Series(0.0, index=df.index)).fillna(0)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.ffill(inplace=True)
    df.dropna(inplace=True)

    return df[FEATURES_V3]


# ── Load model artifacts (cached per ticker) ─────────────────
_model_cache: dict = {}

def get_models(ticker: str):
    if ticker not in _model_cache:
        model_dir = MODEL_BASE / ticker
        if not model_dir.is_dir():
            raise FileNotFoundError(f"No model for {ticker}")

        with open(model_dir / "metadata.json") as f:
            meta = json.load(f)

        scaler = StandardScaler()
        scaler.mean_          = np.load(model_dir / "scaler_mean.npy")
        scaler.scale_         = np.load(model_dir / "scaler_scale.npy")
        scaler.n_features_in_ = len(FEATURES_V3)

        xgb_models = []
        for h in HORIZONS:
            m = xgb.XGBRegressor()
            m.load_model(str(model_dir / f"xgb_{h}.json"))
            xgb_models.append(m)

        deep = DSM9_v3(n_feat=len(FEATURES_V3)).to(DEVICE)
        deep.load_state_dict(load_file(str(model_dir / "model_v3.safetensors")))
        deep.eval()

        _model_cache[ticker] = (scaler, xgb_models, deep, meta)

    return _model_cache[ticker]


# ── Signal helpers ────────────────────────────────────────────
_CLIP_RANGE = {90: 0.25, 365: 0.70}

def signal_bias(ret: float) -> str:
    if ret >  0.20: return "Strong Bullish"
    if ret >  0.05: return "Bullish"
    if ret < -0.20: return "Strong Bearish"
    if ret < -0.05: return "Bearish"
    return "Neutral"

def to_cat(s: str) -> str:
    if "Bullish" in s: return "BULL"
    if "Bearish" in s: return "BEAR"
    return "NEUTRAL"


# ── True walk-forward backtest for one ticker + one window ────
def backtest_ticker_at_date(ticker: str,
                             test_cutoff_days_ago: int,
                             nifty_full: pd.DataFrame,
                             fii_dii: pd.DataFrame):
    """
    Simulates what the model would have predicted `test_cutoff_days_ago`
    days ago, using only data available up to that date.
    """
    csv_path = DATA_CACHE / f"{ticker}.csv"
    if not csv_path.exists():
        return {"ticker": ticker, "error": "CSV not found"}

    raw = pd.read_csv(csv_path, index_col=0, parse_dates=True).sort_index()

    # Need enough rows for: feature warmup + cutoff + 90d outcome
    if len(raw) < test_cutoff_days_ago + 90 + WINDOW + 60:
        return {"ticker": ticker, "error": f"insufficient rows ({len(raw)})"}

    cutoff_idx  = len(raw) - test_cutoff_days_ago
    outcome_idx = cutoff_idx + 90

    if outcome_idx >= len(raw):
        return {"ticker": ticker, "error": "outcome date out of range"}

    price_at_pred    = float(raw["Close"].iloc[cutoff_idx - 1])
    price_at_outcome = float(raw["Close"].iloc[outcome_idx - 1])
    actual_return    = (price_at_outcome - price_at_pred) / price_at_pred
    actual_signal    = signal_bias(actual_return)
    actual_direction = "UP" if actual_return > 0 else "DOWN"

    # Slice everything to cutoff — no future data leakage
    df_slice     = raw.iloc[:cutoff_idx].copy()
    nifty_slice  = nifty_full[nifty_full.index <= df_slice.index[-1]]

    try:
        feat_df = compute_features_from_slice(df_slice, nifty_slice, fii_dii)
    except Exception as e:
        return {"ticker": ticker, "error": f"feature error: {e}"}

    if feat_df is None or len(feat_df) < WINDOW:
        n = len(feat_df) if feat_df is not None else 0
        return {"ticker": ticker, "error": f"too few feature rows ({n})"}

    try:
        scaler, xgb_models, deep_model, meta = get_models(ticker)

        X        = feat_df[FEATURES_V3].values
        X_scaled = scaler.transform(X)

        xgb_preds = np.array([
            float(np.atleast_1d(m.predict(X_scaled[-1:]))[0])
            for m in xgb_models
        ])

        X_seq = torch.tensor(
            X_scaled[-WINDOW:].reshape(1, WINDOW, len(FEATURES_V3)),
            dtype=torch.float32,
        ).to(DEVICE)
        xgb_t = torch.tensor(xgb_preds.reshape(1, -1), dtype=torch.float32).to(DEVICE)

        with torch.no_grad():
            deep_out = deep_model(X_seq, xgb_t).cpu().numpy()[0]

        h_idx    = HORIZONS.index(90) if 90 in HORIZONS else 0
        clip     = _CLIP_RANGE.get(HORIZONS[h_idx], 0.35)
        pred_ret = float(np.clip(deep_out[h_idx], -clip, clip))

        pred_signal    = signal_bias(pred_ret)
        pred_direction = "UP" if pred_ret > 0 else "DOWN"

        return {
            "ticker":            ticker,
            "price_at_pred":     round(price_at_pred, 2),
            "price_at_outcome":  round(price_at_outcome, 2),
            "actual_return_pct": round(actual_return * 100, 2),
            "pred_return_pct":   round(pred_ret * 100, 2),
            "actual_signal":     actual_signal,
            "pred_signal":       pred_signal,
            "direction_correct": pred_direction == actual_direction,
            "signal_correct":    to_cat(pred_signal) == to_cat(actual_signal),
            "return_error_pct":  round(abs(pred_ret - actual_return) * 100, 2),
            "pred_direction":    pred_direction,
            "actual_direction":  actual_direction,
        }

    except Exception as e:
        return {"ticker": ticker, "error": str(e)}


# ── Multi-window backtest ─────────────────────────────────────
def run_full_accuracy_test(cutoff_windows=[100, 150, 200, 250]):
    print("\n" + "="*65)
    print("  DSM-9 MODEL 3.0 — TRUE WALK-FORWARD ACCURACY TEST")
    print("="*65)
    print("  Features: exact 24-feature set from training notebook")
    print("  Method  : historical slice per window (no data leakage)")

    # Load shared data once
    nifty_path = DATA_CACHE / "^NSEI.csv"
    if not nifty_path.exists():
        print("[FAIL] Missing data_cache/^NSEI.csv — required for NIFTY features")
        return
    nifty_full = pd.read_csv(nifty_path, index_col=0, parse_dates=True).sort_index()
    fii_dii    = load_fii_dii()
    print(f"  NIFTY rows: {len(nifty_full)} | FII/DII: {'yes' if not fii_dii.empty else 'no (zeros used)'}")

    all_results    = []
    window_summary = {}

    for window in cutoff_windows:
        print(f"\n[DATE] Testing predictions made ~{window} days ago (90d horizon)...")
        window_results = []

        for ticker in TICKERS:
            r = backtest_ticker_at_date(ticker, window, nifty_full, fii_dii)
            if r and "error" not in r:
                window_results.append(r)
                status = "[OK]" if r["direction_correct"] else "[FAIL]"
                print(f"  {status} {ticker:20} | Pred: {r['pred_return_pct']:+6.1f}% "
                      f"| Actual: {r['actual_return_pct']:+6.1f}% "
                      f"| {r['pred_signal']:15} -> {r['actual_signal']}")
            elif r and "error" in r:
                print(f"  [WARN] {ticker:20} | {r['error'][:60]}")

        if window_results:
            w_dir = sum(1 for r in window_results if r["direction_correct"])
            w_sig = sum(1 for r in window_results if r["signal_correct"])
            w_err = np.mean([r["return_error_pct"] for r in window_results])
            window_summary[str(window)] = {
                "direction_accuracy": round(w_dir / len(window_results) * 100, 2),
                "signal_accuracy":    round(w_sig / len(window_results) * 100, 2),
                "avg_return_error":   round(float(w_err), 2),
                "n":                  len(window_results),
            }
            print(f"\n  --> W{window}: dir={w_dir/len(window_results)*100:.1f}%  "
                  f"sig={w_sig/len(window_results)*100:.1f}%  "
                  f"err={w_err:.1f}%  n={len(window_results)}")

        all_results.extend(window_results)

    # ── Overall metrics ───────────────────────────────────────
    if not all_results:
        print("\n[FAIL] No results — check data_cache/ and model_storage/")
        return

    total               = len(all_results)
    direction_correct   = sum(1 for r in all_results if r["direction_correct"])
    signal_correct      = sum(1 for r in all_results if r["signal_correct"])
    avg_return_error    = np.mean([r["return_error_pct"] for r in all_results])
    median_return_error = np.median([r["return_error_pct"] for r in all_results])

    bullish_preds   = [r for r in all_results if "Bullish" in r["pred_signal"]]
    bearish_preds   = [r for r in all_results if "Bearish" in r["pred_signal"]]
    bullish_correct = sum(1 for r in bullish_preds if r["actual_return_pct"] > 0)
    bearish_correct = sum(1 for r in bearish_preds if r["actual_return_pct"] < 0)

    dir_acc = direction_correct / total * 100

    print("\n" + "="*65)
    print("  [STATS] ACCURACY SUMMARY")
    print("="*65)
    print(f"  Total predictions tested : {total}")
    print(f"  Direction accuracy       : {dir_acc:.1f}%  ({direction_correct}/{total})")
    print(f"  Signal category accuracy : {signal_correct/total*100:.1f}%  ({signal_correct}/{total})")
    print(f"  Avg return error         : {avg_return_error:.1f}%")
    print(f"  Median return error      : {median_return_error:.1f}%")
    if bullish_preds:
        print(f"  Bullish signal accuracy  : {bullish_correct/len(bullish_preds)*100:.1f}%  ({bullish_correct}/{len(bullish_preds)})")
    if bearish_preds:
        print(f"  Bearish signal accuracy  : {bearish_correct/len(bearish_preds)*100:.1f}%  ({bearish_correct}/{len(bearish_preds)})")

    print("\n  Per-window breakdown:")
    for w, s in window_summary.items():
        trend = ""
        if int(w) <= 100: trend = "  <- most recent"
        if int(w) >= 250: trend = "  <- oldest"
        print(f"    {w}d ago: dir={s['direction_accuracy']}%  sig={s['signal_accuracy']}%  err={s['avg_return_error']}%{trend}")

    print("\n" + "="*65)
    print("  [TARGET] VERDICT")
    print("="*65)
    if dir_acc >= 65:
        print(f"  [OK] GOOD — {dir_acc:.1f}% direction accuracy (above 65% threshold)")
        print("       Model has genuine predictive power.")
    elif dir_acc >= 55:
        print(f"  [WARN] MODERATE — {dir_acc:.1f}% direction accuracy")
        print("       Better than random (50%) but needs improvement.")
    else:
        print(f"  [FAIL] POOR — {dir_acc:.1f}% direction accuracy (near random 50%)")
        print("       Model is mostly guessing direction.")

    if avg_return_error > 20:
        print(f"\n  [WARN] Return magnitude error is HIGH ({avg_return_error:.1f}%)")
    print("="*65)

    # ── Save ──────────────────────────────────────────────────
    output = {
        "run_at":              datetime.utcnow().isoformat() + "Z",
        "test_method":         "true_walk_forward_24feat",
        "total_tested":        total,
        "direction_accuracy":  round(dir_acc, 2),
        "signal_accuracy":     round(signal_correct / total * 100, 2),
        "avg_return_error":    round(float(avg_return_error), 2),
        "median_return_error": round(float(median_return_error), 2),
        "bullish_accuracy":    round(bullish_correct / len(bullish_preds) * 100, 2) if bullish_preds else None,
        "bearish_accuracy":    round(bearish_correct / len(bearish_preds) * 100, 2) if bearish_preds else None,
        "window_summary":      window_summary,
        "details":             all_results,
    }

    out_path = Path("accuracy_report.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  [FILE] Full results saved -> {out_path}")
    return output


if __name__ == "__main__":
    run_full_accuracy_test(cutoff_windows=[100, 150, 200, 250])