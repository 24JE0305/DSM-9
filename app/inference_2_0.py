from fastapi import APIRouter
from model_2.production_predictor_2_0 import ProductionPredictor2_0
import numpy as np

router = APIRouter()
predictor = ProductionPredictor2_0()


@router.post("/predict_v2")
def predict(symbol: str):

    # TODO: Replace with real feature builder
    dummy_90 = np.random.rand(1, 50)
    dummy_365 = np.random.rand(1, 50)

    result = predictor.predict(
        symbol=symbol,
        features_90=dummy_90,
        features_365=dummy_365
    )

    return result
