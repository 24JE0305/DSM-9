from pydantic import BaseModel
from typing import Dict


class PredictionRequest(BaseModel):
    ticker: str


class PredictionResponse(BaseModel):
    ticker: str
    last_close: float
    predictions: Dict[str, float]
