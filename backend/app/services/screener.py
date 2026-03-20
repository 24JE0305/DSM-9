# ============================================================
# DSM-9  STOCK SCREENER SERVICE
# app/services/screener.py
#
# Fix vs old version: per-signal caps (MAX_BULLISH=30, MAX_BEARISH=10)
# ============================================================

import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

from app.config import TOP50_FILE
from app.model_3.inference_v3 import predict_v3


_PREDICT_CACHE = {}
CACHE_TTL = 3600  # 1 hour

def _safe_predict(ticker: str) -> Optional[dict]:
    now = time.time()
    if ticker in _PREDICT_CACHE:
        cached_result, timestamp = _PREDICT_CACHE[ticker]
        if now - timestamp < CACHE_TTL:
            return cached_result
            
    try:
        res = predict_v3(ticker)
        _PREDICT_CACHE[ticker] = (res, now)
        return res
    except Exception as e:
        return {"symbol": ticker, "_error": str(e)}


def _load_tickers() -> list:
    with open(TOP50_FILE) as f:
        return json.load(f)["tickers"]


def _matches(value: str, filter_str: Optional[str]) -> bool:
    if not filter_str:
        return True
    allowed = [s.strip().lower() for s in filter_str.split(",")]
    return value.lower() in allowed


def _horizon_key(horizon: int) -> str:
    return f"{horizon}_days"


def run_screener(
    horizon: int = 90,
    signal: Optional[str] = None,
    confidence: Optional[str] = None,
    min_return: Optional[float] = None,
    max_return: Optional[float] = None,
    min_agreement: Optional[float] = None,
    volatility: Optional[str] = None,
    sort_by: str = "return",
    limit: int = 50,
) -> dict:

    if horizon not in (90, 365):
        raise ValueError("horizon must be 90 or 365")

    tickers = _load_tickers()
    h_key   = _horizon_key(horizon)

    with ThreadPoolExecutor(max_workers=8) as pool:
        raw = list(pool.map(_safe_predict, tickers))

    results = []
    errors  = []

    for r in raw:
        if r is None:
            continue
        if "_error" in r:
            errors.append({"symbol": r["symbol"], "error": r["_error"]})
            continue

        preds = r.get("predictions", {})
        if h_key not in preds:
            continue

        p          = preds[h_key]
        risk       = r.get("risk_metrics", {})
        er         = p.get("expected_return", 0)
        return_pct = round(er * 100, 2)
        conf_score = p.get("confidence_score", 0)
        conf_level = p.get("confidence_level", "Low")
        sig_bias   = p.get("signal_bias", "Neutral")
        agreement  = risk.get("model_agreement", 0)
        vol_level  = risk.get("volatility_level", "Low")
        vol_ratio  = risk.get("volatility_ratio", 0)
        pred_range = p.get("range", {})

        if not _matches(sig_bias,    signal):      continue
        if not _matches(conf_level,  confidence):  continue
        if not _matches(vol_level,   volatility):  continue
        if min_return    is not None and return_pct < min_return:   continue
        if max_return    is not None and return_pct > max_return:   continue
        if min_agreement is not None and agreement < min_agreement: continue

        results.append({
            "symbol":           r["symbol"],
            "signal_bias":      sig_bias,
            "expected_return":  return_pct,
            "confidence_score": round(conf_score, 2),
            "confidence_level": conf_level,
            "model_agreement":  round(agreement, 2),
            "volatility_level": vol_level,
            "volatility_ratio": round(vol_ratio, 4),
            "range": {
                "low":      round(pred_range.get("low",  0) * 100, 2),
                "expected": return_pct,
                "high":     round(pred_range.get("high", 0) * 100, 2),
            },
            "reliability": r.get("reliability", {}),
            "horizon_days": horizon,
            "generated_at": r.get("generated_at"),
        })

    # ── Sort ─────────────────────────────────────────────────
    sort_map = {
        "return":     lambda x: x["expected_return"],
        "confidence": lambda x: x["confidence_score"],
        "agreement":  lambda x: x["model_agreement"],
    }
    sort_fn = sort_map.get(sort_by, sort_map["return"])
    results.sort(key=sort_fn, reverse=True)

    # ── Per-signal caps ───────────────────────────────────────
    MAX_BULLISH = 30   # Strong Bullish + Bullish combined
    MAX_BEARISH = 10   # Strong Bearish + Bearish combined

    bullish_bucket = [r for r in results if "Bullish" in r["signal_bias"]][:MAX_BULLISH]
    bearish_bucket = [r for r in results if "Bearish" in r["signal_bias"]][:MAX_BEARISH]
    neutral_bucket = [r for r in results if r["signal_bias"] == "Neutral"]

    results = sorted(
        bullish_bucket + bearish_bucket + neutral_bucket,
        key=sort_fn, reverse=True,
    )[:limit]

    # ── Summary ───────────────────────────────────────────────
    total         = len(results)
    bullish_count = sum(1 for r in results if "Bullish" in r["signal_bias"])
    bearish_count = sum(1 for r in results if "Bearish" in r["signal_bias"])
    avg_return    = round(sum(r["expected_return"]  for r in results) / total, 2) if total else 0
    avg_conf      = round(sum(r["confidence_score"] for r in results) / total, 2) if total else 0
    avg_agreement = round(sum(r["model_agreement"]  for r in results) / total, 2) if total else 0

    return {
        "total_matched":  total,
        "total_screened": len(tickers),
        "failed":         len(errors),
        "summary": {
            "bullish_count":  bullish_count,
            "bearish_count":  bearish_count,
            "neutral_count":  total - bullish_count - bearish_count,
            "avg_return_pct": avg_return,
            "avg_confidence": avg_conf,
            "avg_agreement":  avg_agreement,
            "market_mood": (
                "Bullish" if bullish_count > bearish_count
                else "Bearish" if bearish_count > bullish_count
                else "Neutral"
            ),
        },
        "filters_applied": {
            "horizon":       horizon,
            "signal":        signal      or "all",
            "confidence":    confidence  or "all",
            "min_return":    min_return,
            "max_return":    max_return,
            "min_agreement": min_agreement,
            "volatility":    volatility  or "all",
            "sort_by":       sort_by,
            "limit":         limit,
        },
        "results":    results,
        "errors":     errors,
        "disclaimer": "AI-generated probabilistic forecast. Not financial advice.",
    }