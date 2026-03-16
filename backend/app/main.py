from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.schemas import PredictionRequest, PredictionResponse
from app.inference import predict_ticker
import json
import pandas as pd
from app.services.backtest_service import run_backtest
import os
from app.config import TOP50_FILE
from app.model_3.inference_v3 import predict_v3
from app.services.backtest_v3 import run_backtest_v3

# --- Model 2 Imports ---
from app.model_2.inference_v2 import predict_v2


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


# ==========================================
#          MODEL 1.0 ENDPOINTS
# ==========================================

@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest):
    try:
        return predict_ticker(req.ticker)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ==========================================
#          MODEL 2.0 ENDPOINTS
# ==========================================

@app.get("/predict_v2/{ticker}")
def predict_model_2(ticker: str):
    try:
        result = predict_v2(ticker)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    
'''
Component	Famous Equation	Purpose
Growth Rating	Sharpe Ratio	Risk-adjusted growth
Financial Health	Piotroski F-Score	Fundamental strength
Safety Rating	Altman Z-Score	Bankruptcy risk
'''
# ==========================================
#          BACKTEST ENDPOINTS
# ==========================================
# Add this import at the top of app/main.py:
#   from app.services.backtest_service import run_backtest

@app.get("/backtest_v2/{ticker}")
def backtest_model_2(
    ticker: str,
    horizon: int = 90,   # ?horizon=90
    step: int = 30,      # ?step=30
):
    """
    Walk-forward backtest for Model 2 on a single ticker.

    Query params
    ------------
    horizon : int  — How many days each trade is held (default 90)
    step    : int  — How often the model re-evaluates in rows (default 30)

    Example
    -------
    GET /backtest_v2/ADANIENT.NS
    GET /backtest_v2/ADANIENT.NS?horizon=90&step=30
    """
    try:
        result = run_backtest(ticker, horizon=horizon, step=step)
        return result
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
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
        result = run_backtest_v3(ticker, horizon=horizon, step=step)
        return result
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Side-by-side model comparison ───────────────────────────

@app.get("/compare/{ticker}")
def compare_models(ticker: str):
    """
    Run both Model 2 and Model 3 and return side-by-side predictions.
    Useful for evaluating upgrade quality.
    """
    result = {"ticker": ticker, "generated_at": __import__('datetime').datetime.utcnow().isoformat() + "Z"}

    try:
        result["model_2"] = predict_v2(ticker)
    except Exception as e:
        result["model_2"] = {"error": str(e)}

    try:
        result["model_3"] = predict_v3(ticker)
    except Exception as e:
        result["model_3"] = {"error": str(e)}

    return result