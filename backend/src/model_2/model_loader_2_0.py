import os
import json
import xgboost as xgb
import torch
from safetensors.torch import load_file


class ModelLoader2_0:
    def __init__(self, base_path="model_storage/model_2"):
        self.base_path = base_path

    def _get_symbol_path(self, symbol):
        return os.path.join(self.base_path, symbol)

    def load_xgb(self, symbol, horizon):
        model_path = os.path.join(
            self._get_symbol_path(symbol),
            f"xgb_{horizon}.json"
        )

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"XGB model not found: {model_path}")

        model = xgb.XGBRegressor()
        model.load_model(model_path)
        return model

    def load_lstm(self, symbol):
        model_path = os.path.join(
            self._get_symbol_path(symbol),
            "lstm.safetensors"
        )

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"LSTM model not found: {model_path}")

        weights = load_file(model_path)
        return weights
