from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.model_2.inference_v2 import predict_v2

app = FastAPI(title="DSM-9 Market Prediction API v2")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def health():
    return {"status": "ok", "model_version": "2.0"}


@app.get("/predict_v2/{ticker}")
def predict(ticker: str):

    try:
        result = predict_v2(ticker)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))