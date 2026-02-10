from fastapi import FastAPI, HTTPException
from app.schemas import PredictionRequest, PredictionResponse
from app.inference import predict_ticker
import json
from app.config import TOP50_FILE

app = FastAPI(title="DSM-9 Market Prediction API")


@app.get("/")
def health():
    return {"status": "ok"}


@app.get("/tickers")
def get_top50():
    with open(TOP50_FILE) as f:
        return json.load(f)


@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest):
    try:
        return predict_ticker(req.ticker)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
