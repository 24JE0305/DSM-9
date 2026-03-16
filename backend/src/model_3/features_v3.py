# ============================================================
# DSM-9  MODEL 3.0 — FEATURE ENGINEERING
# src/model_3/features_v3.py
# ============================================================

import numpy as np
import pandas as pd
from pathlib import Path

DATA_CACHE = Path("data_cache")

# ── All features produced by this module ────────────────────
FEATURES_V3 = [
    # Price / return
    "Close", "Return", "Log_Return",
    # Trend
    "MA10", "MA50", "MA200", "EMA20",
    "Price_vs_MA50", "Price_vs_MA200",
    # Momentum
    "RSI", "MACD", "MACD_Signal", "MACD_Hist",
    # Volatility
    "BB_Upper", "BB_Lower", "BB_Width", "ATR",
    # Volume
    "OBV", "Volume_Change", "Volume_MA20",
    # Market context
    "NIFTY_Return", "NIFTY_MA10",
    # FII / DII  (filled with 0 when unavailable)
    "FII_Net", "DII_Net",
]


# ── Individual indicator helpers ─────────────────────────────

def compute_rsi(s: pd.Series, period: int = 14) -> pd.Series:
    d = s.diff()
    g = d.where(d > 0, 0).rolling(period).mean()
    l = (-d.where(d < 0, 0)).rolling(period).mean()
    return 100 - 100 / (1 + g / (l + 1e-9))


def compute_macd(s: pd.Series, fast=12, slow=26, signal=9):
    ema_fast   = s.ewm(span=fast,   adjust=False).mean()
    ema_slow   = s.ewm(span=slow,   adjust=False).mean()
    macd_line  = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist        = macd_line - signal_line
    return macd_line, signal_line, hist


def compute_bollinger(s: pd.Series, period=20, n_std=2):
    ma     = s.rolling(period).mean()
    std    = s.rolling(period).std()
    upper  = ma + n_std * std
    lower  = ma - n_std * std
    width  = (upper - lower) / (ma + 1e-9)
    return upper, lower, width


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period=14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def compute_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff().fillna(0))
    return (direction * volume).cumsum()


# ── FII / DII loader ─────────────────────────────────────────
# Expects data_cache/FII_DII.csv with columns: Date, FII_Net, DII_Net
# If file missing → fills zeros (model still works, just without that signal)

def load_fii_dii() -> pd.DataFrame:
    path = DATA_CACHE / "FII_DII.csv"
    if not path.exists():
        return pd.DataFrame(columns=["FII_Net", "DII_Net"])
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df = df[["FII_Net", "DII_Net"]].copy()
    df["FII_Net"] = pd.to_numeric(df["FII_Net"], errors="coerce").fillna(0)
    df["DII_Net"] = pd.to_numeric(df["DII_Net"], errors="coerce").fillna(0)
    return df


# ── Main feature builder ─────────────────────────────────────

def compute_features_v3(ticker: str) -> pd.DataFrame:
    """
    Build the full Model 3.0 feature matrix for a ticker.
    Returns a clean DataFrame with columns = FEATURES_V3.
    Raises FileNotFoundError if cache CSVs are missing.
    """
    ticker_path = DATA_CACHE / f"{ticker}.csv"
    nifty_path  = DATA_CACHE / "^NSEI.csv"

    if not ticker_path.exists():
        raise FileNotFoundError(f"Cache missing: {ticker_path}")
    if not nifty_path.exists():
        raise FileNotFoundError("NIFTY cache missing: data_cache/^NSEI.csv")

    df    = pd.read_csv(ticker_path, index_col=0, parse_dates=True)
    nifty = pd.read_csv(nifty_path,  index_col=0, parse_dates=True)

    # ── Require OHLCV ────────────────────────────────────────
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' in {ticker_path}")

    # ── Price / return ───────────────────────────────────────
    df["Return"]     = df["Close"].pct_change()
    df["Log_Return"] = np.log(df["Close"] / df["Close"].shift(1))

    # ── Trend ────────────────────────────────────────────────
    df["MA10"]  = df["Close"].rolling(10).mean()
    df["MA50"]  = df["Close"].rolling(50).mean()
    df["MA200"] = df["Close"].rolling(200).mean()
    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["Price_vs_MA50"]  = (df["Close"] - df["MA50"])  / (df["MA50"]  + 1e-9)
    df["Price_vs_MA200"] = (df["Close"] - df["MA200"]) / (df["MA200"] + 1e-9)

    # ── Momentum ─────────────────────────────────────────────
    df["RSI"] = compute_rsi(df["Close"])
    macd, macd_sig, macd_hist = compute_macd(df["Close"])
    df["MACD"]        = macd
    df["MACD_Signal"] = macd_sig
    df["MACD_Hist"]   = macd_hist

    # ── Volatility ───────────────────────────────────────────
    bb_u, bb_l, bb_w = compute_bollinger(df["Close"])
    df["BB_Upper"] = bb_u
    df["BB_Lower"] = bb_l
    df["BB_Width"] = bb_w
    df["ATR"]      = compute_atr(df["High"], df["Low"], df["Close"])

    # ── Volume ───────────────────────────────────────────────
    df["OBV"]          = compute_obv(df["Close"], df["Volume"])
    df["Volume_Change"] = df["Volume"].pct_change()
    df["Volume_MA20"]  = df["Volume"].rolling(20).mean()

    # ── Market context (NIFTY) ───────────────────────────────
    nifty["NIFTY_Return"] = nifty["Close"].pct_change()
    nifty["NIFTY_MA10"]   = nifty["Close"].rolling(10).mean()
    df = df.join(nifty[["NIFTY_Return", "NIFTY_MA10"]], how="left")

    # ── FII / DII ────────────────────────────────────────────
    fii_dii = load_fii_dii()
    if not fii_dii.empty:
        df = df.join(fii_dii, how="left")
        df["FII_Net"] = df["FII_Net"].fillna(0)
        df["DII_Net"] = df["DII_Net"].fillna(0)
    else:
        df["FII_Net"] = 0.0
        df["DII_Net"] = 0.0

    # ── Clean ────────────────────────────────────────────────
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.ffill(inplace=True)
    df.dropna(inplace=True)

    # Validate all features exist
    missing = [f for f in FEATURES_V3 if f not in df.columns]
    if missing:
        raise ValueError(f"Missing features after engineering: {missing}")

    print(f"[{ticker}] Feature shape: {df[FEATURES_V3].shape}")
    return df[FEATURES_V3]
