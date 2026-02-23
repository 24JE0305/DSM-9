from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.schemas import PredictionRequest, PredictionResponse
from app.inference import predict_ticker
import json
import pandas as pd
import os
from app.config import TOP50_FILE

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