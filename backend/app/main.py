from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.services.screener import run_screener
from app.services.explainer import explain_ticker
import json
import pandas as pd
import os
from app.config import TOP50_FILE
from app.model_3.inference_v3 import predict_v3
from app.services.backtest_v3 import run_backtest_v3_0_1



app = FastAPI(title="DSM-9 Market Prediction API (Combined)")

# Allow CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def health():
    return {"status": "ok"}


@app.get("/tickers")
def get_top50():
    with open(TOP50_FILE) as f:
        return json.load(f)


@app.get("/history/{ticker}")
def get_history(ticker: str):
    file_path = f"data_cache/{ticker}.csv"
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="History not found")
    
    try:
        df = pd.read_csv(file_path)
        # Convert to list of dicts for JSON response
        # Ensure we have Date, Open, High, Low, Close
        if 'Date' not in df.columns and df.index.name == 'Date':
             df.reset_index(inplace=True)
             
        return df.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================
# MODEL 3.0 ENDPOINTS  — paste into app/main.py
# ============================================================
#
# Add these imports at the top of app/main.py:
#
#   from app.model_3.inference_v3   import predict_v3
#   from app.services.backtest_v3   import run_backtest_v3
#
# ============================================================

# ── Predict ─────────────────────────────────────────────────

@app.get("/predict_v3/{ticker}")
def predict_model_3(ticker: str):
    """
    Model 3.0 prediction — Transformer + BiLSTM + XGBoost + Learned Fusion.
    Returns same JSON structure as /predict_v2 for easy comparison.
    """
    try:
        return predict_v3(ticker)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ── Backtest ────────────────────────────────────────────────

@app.get("/backtest_v3/{ticker}")
def backtest_model_3(ticker: str, horizon: int = 90, step: int = 30):
    """
    Walk-forward backtest using Model 3.0.
    Same metrics as /backtest_v2 for easy A/B comparison.
    """
    try:
        result = run_backtest_v3_0_1(ticker, horizon=horizon, step=step)
        return result
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================
# ADD THIS TO app/main.py
# ============================================================

# Add this import at the top of main.py:
#   from app.services.screener import run_screener

# ── Screener Endpoint ────────────────────────────────────────

@app.get("/screener")
def stock_screener(
    horizon: int = 90,
    signal: str = None,           # "Bullish" or "Bullish,Strong Bullish"
    confidence: str = None,       # "High" or "High,Moderate"
    min_return: float = None,     # 5.0 means 5%
    max_return: float = None,     # 50.0 means 50%
    min_agreement: float = None,  # 0.7
    volatility: str = None,       # "Low" or "Low,Moderate"
    sort_by: str = "return",      # return | confidence | agreement
    limit: int = 50,
):
    """
    Screen all Nifty Top-50 stocks using Model 3.0 predictions.

    Examples
    --------
    # All bullish stocks with high confidence, sorted by return
    GET /screener?signal=Bullish,Strong Bullish&confidence=High&sort_by=return

    # Stocks with >5% expected return in 90 days, high model agreement
    GET /screener?min_return=5&min_agreement=0.7&horizon=90

    # Best 10 long-term (365d) picks
    GET /screener?horizon=365&sort_by=return&limit=10

    # Conservative picks — high confidence, low volatility
    GET /screener?confidence=High&volatility=Low&sort_by=confidence
    """
    try:
        result = run_screener(
            horizon=horizon,
            signal=signal,
            confidence=confidence,
            min_return=min_return,
            max_return=max_return,
            min_agreement=min_agreement,
            volatility=volatility,
            sort_by=sort_by,
            limit=limit,
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================
# ADD THIS TO app/main.py
# ============================================================

# Add this import at the top of main.py:
#   from app.services.explainer import explain_ticker

# ── Explainer Endpoint ───────────────────────────────────────

@app.get("/explain/{ticker}")
def explain_stock(ticker: str):
    """
    Explain why a stock is predicted Bullish or Bearish.
    Uses XGBoost feature importance (gain) to show the top
    drivers behind the prediction with human-readable context.

    Examples
    --------
    GET /explain/RELIANCE.NS
    GET /explain/TCS.NS
    GET /explain/HDFCBANK.NS
    """
    try:
        return explain_ticker(ticker)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))