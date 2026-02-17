import numpy as np
from src.model_loader_2_0 import ModelLoader2_0
from src.fusion_engine_2_0 import FusionEngine2_0


class ProductionPredictor2_0:

    def __init__(self):
        self.loader = ModelLoader2_0()
        self.fusion = FusionEngine2_0()

    def predict(self, symbol, features_90, features_365):

        xgb_90 = self.loader.load_xgb(symbol, 90)
        xgb_365 = self.loader.load_xgb(symbol, 365)

        pred_90 = xgb_90.predict(features_90)[0]
        pred_365 = xgb_365.predict(features_365)[0]

        fusion_result = self.fusion.fuse({
            90: pred_90,
            365: pred_365
        })

        return {
            "symbol": symbol,
            "horizon_predictions": {
                "90_days": float(pred_90),
                "365_days": float(pred_365)
            },
            "ultimate_prediction": fusion_result["fused_prediction"],
            "confidence": fusion_result["confidence_score"]
        }
