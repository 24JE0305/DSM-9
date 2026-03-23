# ============================================================
# DSM-9  MODEL 3.0 — MONGODB LOADER
# app/model_3/mongo_loader.py
#
# Loads model artifacts from MongoDB (GridFS for binary files,
# documents for JSON files).  Provides the same return signature
# as the local _load_v3_models() so inference_v3.py needs only
# a one-line swap.
# ============================================================

import io
import json
import numpy as np
import torch
import xgboost as xgb
from functools import lru_cache
from safetensors.torch import load_file as safetensors_load_file
from sklearn.preprocessing import StandardScaler
import gridfs
from pymongo import MongoClient
from app.config import MONGODB_URI, MONGODB_DB

from src.model_3.model_v3_0 import DSM9_v3, DEVICE, HORIZONS
from src.model_3.features_v3 import FEATURES_V3


# ── Singleton MongoDB client ──────────────────────────────────

_client: MongoClient = None
_db = None
_fs: gridfs.GridFS = None


def _get_db():
    global _client, _db, _fs
    if _client is None:
        _client = MongoClient(MONGODB_URI)
        _db = _client[MONGODB_DB]
        _fs = gridfs.GridFS(_db)
    return _db, _fs


# ── GridFS helpers ────────────────────────────────────────────

def _read_gridfs_bytes(fs: gridfs.GridFS, filename: str) -> bytes:
    """Read a file from GridFS by its stored filename."""
    f = fs.find_one({"filename": filename})
    if f is None:
        raise FileNotFoundError(f"GridFS file not found: {filename}")
    return f.read()


# ── Cached loader ─────────────────────────────────────────────

@lru_cache(maxsize=60)
def load_v3_models_from_mongo(ticker: str):
    """
    Load Model 3.0 artifacts for *ticker* from MongoDB.

    Returns
    -------
    (scaler, xgb_models, deep_model, metadata)
      — same tuple as the old _load_v3_models() for drop-in replacement.
    """
    db, fs = _get_db()

    # ── metadata.json  (stored as a regular document) ─────────
    doc = db["models"].find_one({"ticker": ticker, "file": "metadata.json"})
    if doc is None:
        raise FileNotFoundError(
            f"No Model 3.0 found in MongoDB for {ticker}. Run upload_models_to_mongo.py first."
        )
    metadata = doc["content"]

    # ── Scaler (numpy arrays from GridFS) ────────────────────
    scaler = StandardScaler()
    scaler.mean_ = np.load(
        io.BytesIO(_read_gridfs_bytes(fs, f"{ticker}/scaler_mean.npy"))
    )
    scaler.scale_ = np.load(
        io.BytesIO(_read_gridfs_bytes(fs, f"{ticker}/scaler_scale.npy"))
    )
    scaler.n_features_in_ = len(FEATURES_V3)

    # ── XGBoost models (JSON bytes from GridFS) ───────────────
    xgb_models = []
    for h in HORIZONS:
        xgb_bytes = _read_gridfs_bytes(fs, f"{ticker}/xgb_{h}.json")
        m = xgb.XGBRegressor()
        # XGBoost can load from a temporary file or a Booster object;
        # write to an in-memory temp path via BytesIO trick with load_model
        m.load_model(bytearray(xgb_bytes))
        xgb_models.append(m)

    # ── Deep model (safetensors from GridFS) ──────────────────
    safetensors_bytes = _read_gridfs_bytes(
        fs, f"{ticker}/model_v3.safetensors"
    )
    # safetensors requires a file path; write to a BytesIO-compatible buffer
    buf = io.BytesIO(safetensors_bytes)
    # Use the memory-mapped approach: load from bytes directly
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as tmp:
        tmp.write(safetensors_bytes)
        tmp_path = tmp.name
    try:
        deep_model = DSM9_v3(n_feat=len(FEATURES_V3)).to(DEVICE)
        state_dict = safetensors_load_file(tmp_path)
        deep_model.load_state_dict(state_dict)
        deep_model.eval()
    finally:
        os.unlink(tmp_path)

    return scaler, xgb_models, deep_model, metadata
