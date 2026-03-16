import numpy as np


class FusionEngine2_0:

    def __init__(self):
        # Stronger weight for long term stability
        self.weights = {
            90: 0.4,
            365: 0.6
        }

    def fuse(self, preds):
        """
        preds = {
            90: value,
            365: value
        }
        """

        weighted_sum = 0
        total_weight = 0

        for horizon, value in preds.items():
            w = self.weights.get(horizon, 0)
            weighted_sum += value * w
            total_weight += w

        fused = weighted_sum / total_weight

        # Confidence = agreement score
        disagreement = np.std(list(preds.values()))
        confidence = 1 / (1 + disagreement)

        return {
            "fused_prediction": float(fused),
            "confidence_score": float(confidence)
        }
