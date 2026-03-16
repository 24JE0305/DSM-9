# ============================================================
# DSM-9  FEATURE IMPORTANCE — EXPLAINER SERVICE
# app/services/explainer.py
#
# Uses XGBoost's built-in feature importance to explain
# why a stock is predicted Bullish/Bearish.
# No new dependencies — XGBoost already installed.
# ============================================================

import json
import numpy as np
import xgboost as xgb
from pathlib import Path
from sklearn.preprocessing import StandardScaler

from src.model_3.features_v3 import compute_features_v3, FEATURES_V3
from src.model_3.model_v3_0 import HORIZONS

MODEL_BASE = Path("model_storage/model_3")

# ── Human-readable interpretations per feature ───────────────
# Maps feature name → interpretation based on signal direction
_FEATURE_CONTEXT = {
    "RSI": {
        "bullish":  "RSI indicates oversold conditions — potential rebound likely",
        "bearish":  "RSI indicates overbought conditions — pullback risk",
        "neutral":  "RSI is in neutral zone",
    },
    "MACD": {
        "bullish":  "MACD shows bullish momentum building",
        "bearish":  "MACD shows bearish momentum",
        "neutral":  "MACD is flat — no strong momentum signal",
    },
    "MACD_Hist": {
        "bullish":  "MACD histogram turning positive — buyers gaining control",
        "bearish":  "MACD histogram negative — sellers in control",
        "neutral":  "MACD histogram near zero",
    },
    "MACD_Signal": {
        "bullish":  "Price crossing above MACD signal line",
        "bearish":  "Price crossing below MACD signal line",
        "neutral":  "MACD signal line flat",
    },
    "Price_vs_MA50": {
        "bullish":  "Trading above 50-day average — uptrend intact",
        "bearish":  "Trading below 50-day average — downtrend pressure",
        "neutral":  "Trading near 50-day average",
    },
    "Price_vs_MA200": {
        "bullish":  "Above 200-day average — long-term bull trend",
        "bearish":  "Below 200-day average — long-term bear pressure",
        "neutral":  "Near 200-day average — trend undecided",
    },
    "BB_Width": {
        "bullish":  "Bollinger Bands squeezing — breakout likely",
        "bearish":  "Bollinger Bands wide — high volatility, caution",
        "neutral":  "Bollinger Bands normal width",
    },
    "ATR": {
        "bullish":  "Low ATR — stable price action, lower risk entry",
        "bearish":  "High ATR — elevated volatility, higher risk",
        "neutral":  "ATR at normal levels",
    },
    "NIFTY_Return": {
        "bullish":  "Broader market (NIFTY) showing positive momentum",
        "bearish":  "Broader market (NIFTY) under pressure",
        "neutral":  "Broader market (NIFTY) flat",
    },
    "OBV": {
        "bullish":  "On-Balance Volume rising — institutional buying detected",
        "bearish":  "On-Balance Volume falling — distribution phase",
        "neutral":  "OBV flat — no strong volume signal",
    },
    "Volume_Change": {
        "bullish":  "Volume surge — strong buying interest",
        "bearish":  "Volume surge on down days — selling pressure",
        "neutral":  "Volume near average levels",
    },
    "Return": {
        "bullish":  "Recent daily returns positive — short-term momentum",
        "bearish":  "Recent daily returns negative — short-term weakness",
        "neutral":  "Recent returns mixed",
    },
    "MA10": {
        "bullish":  "Short-term trend (MA10) is rising",
        "bearish":  "Short-term trend (MA10) is falling",
        "neutral":  "Short-term trend flat",
    },
    "MA50": {
        "bullish":  "Medium-term trend (MA50) is rising",
        "bearish":  "Medium-term trend (MA50) is falling",
        "neutral":  "Medium-term trend flat",
    },
    "EMA20": {
        "bullish":  "EMA20 trending up — momentum positive",
        "bearish":  "EMA20 trending down — momentum negative",
        "neutral":  "EMA20 flat",
    },
    "FII_Net": {
        "bullish":  "Foreign institutional investors are net buyers",
        "bearish":  "Foreign institutional investors are net sellers",
        "neutral":  "FII activity neutral",
    },
    "DII_Net": {
        "bullish":  "Domestic institutions are net buyers — strong support",
        "bearish":  "Domestic institutions are net sellers",
        "neutral":  "DII activity neutral",
    },
    "NIFTY_MA10": {
        "bullish":  "Market short-term trend supportive",
        "bearish":  "Market short-term trend weak",
        "neutral":  "Market trend neutral",
    },
}

_DEFAULT_CONTEXT = {
    "bullish": "Contributing positively to the bullish outlook",
    "bearish": "Contributing to the bearish signal",
    "neutral": "Minor influence on prediction",
}


def _get_interpretation(feature: str, signal: str) -> str:
    ctx = _FEATURE_CONTEXT.get(feature, _DEFAULT_CONTEXT)
    direction = "bullish" if "bullish" in signal.lower() else \
                "bearish" if "bearish" in signal.lower() else "neutral"
    return ctx.get(direction, ctx.get("neutral", ""))


# ── Core explainer ────────────────────────────────────────────

def explain_ticker(ticker: str) -> dict:
    """
    Explain why a stock is predicted Bullish/Bearish.

    Returns top feature importances from XGBoost for both
    90-day and 365-day horizons with human-readable context.
    """

    # ── Validate ─────────────────────────────────────────────
    model_dir = MODEL_BASE / ticker
    if not model_dir.is_dir():
        raise FileNotFoundError(f"No Model 3.0 found for {ticker}. Train first.")

    # ── Load metadata ────────────────────────────────────────
    with open(model_dir / "metadata.json") as f:
        metadata = json.load(f)

    # ── Load features ────────────────────────────────────────
    df = compute_features_v3(ticker)
    if df is None or df.empty:
        raise ValueError("Feature dataframe is empty")

    X = df[FEATURES_V3].values

    # Scale using saved scaler params
    scaler = StandardScaler()
    scaler.mean_          = np.load(model_dir / "scaler_mean.npy")
    scaler.scale_         = np.load(model_dir / "scaler_scale.npy")
    scaler.n_features_in_ = len(FEATURES_V3)
    X_scaled = scaler.transform(X)
    last_row  = X_scaled[-1:]

    # ── Load XGB models + get importance ─────────────────────
    horizon_explanations = {}

    for i, h in enumerate(HORIZONS):
        model = xgb.XGBRegressor()
        model.load_model(str(model_dir / f"xgb_{h}.json"))

        # Predicted return for this horizon
        pred_return = float(model.predict(last_row)[0])
        signal      = (
            "Strong Bullish" if pred_return >  0.15 else
            "Bullish"        if pred_return >  0    else
            "Strong Bearish" if pred_return < -0.15 else
            "Bearish"        if pred_return <  0    else
            "Neutral"
        )

        # Feature importance scores (gain = most reliable type)
        importance_dict = model.get_booster().get_score(importance_type="gain")

        # Map f0, f1... back to actual feature names
        named_importance = {}
        for fname, score in importance_dict.items():
            if fname.startswith("f"):
                try:
                    idx = int(fname[1:])
                    if idx < len(FEATURES_V3):
                        named_importance[FEATURES_V3[idx]] = float(score)
                except ValueError:
                    named_importance[fname] = float(score)
            else:
                named_importance[fname] = float(score)

        # Normalize to percentages
        total = sum(named_importance.values()) + 1e-9
        normalized = {k: round(v / total, 4) for k, v in named_importance.items()}

        # Sort by importance descending
        sorted_features = sorted(normalized.items(), key=lambda x: x[1], reverse=True)

        # Top 6 features with interpretation
        top_features = []
        for feat, imp in sorted_features[:6]:
            top_features.append({
                "feature":        feat,
                "importance":     round(imp * 100, 1),   # as percentage
                "interpretation": _get_interpretation(feat, signal),
            })

        # Current feature values for context
        last_values = df[FEATURES_V3].iloc[-1]
        feature_snapshot = {
            "RSI":            round(float(last_values.get("RSI", 0)), 1),
            "MACD_Hist":      round(float(last_values.get("MACD_Hist", 0)), 4),
            "Price_vs_MA50":  round(float(last_values.get("Price_vs_MA50", 0)) * 100, 2),
            "Price_vs_MA200": round(float(last_values.get("Price_vs_MA200", 0)) * 100, 2),
            "BB_Width":       round(float(last_values.get("BB_Width", 0)), 4),
            "ATR":            round(float(last_values.get("ATR", 0)), 2),
            "OBV_trend":      "Rising" if float(last_values.get("OBV", 0)) > 0 else "Falling",
            "NIFTY_Return":   round(float(last_values.get("NIFTY_Return", 0)) * 100, 2),
        }

        horizon_explanations[f"{h}_days"] = {
            "predicted_return_pct": round(pred_return * 100, 2),
            "signal":               signal,
            "top_drivers":          top_features,
            "total_features_used":  len(named_importance),
        }

    # ── Summary explanation ───────────────────────────────────
    # Use 90-day signal as primary
    primary = horizon_explanations[f"{HORIZONS[0]}_days"]
    primary_signal = primary["signal"]

    # One-line summary
    top_driver = primary["top_drivers"][0]["feature"] if primary["top_drivers"] else "multiple factors"
    summary = (
        f"{ticker.replace('.NS','')} is {primary_signal} primarily due to "
        f"{top_driver} signals, supported by "
        f"{primary['top_drivers'][1]['feature'] if len(primary['top_drivers']) > 1 else 'technical indicators'}."
    )

    return {
        "symbol":           ticker,
        "summary":          summary,
        "primary_signal":   primary_signal,
        "feature_snapshot": feature_snapshot,
        "horizons":         horizon_explanations,
        "model_info": {
            "version":       metadata.get("version", "3.0"),
            "n_features":    len(FEATURES_V3),
            "importance_type": "gain",
            "note": "Importance scores show which features XGBoost relied on most for this prediction",
        },
        "disclaimer": "AI-generated analysis. Not financial advice.",
    }