# ============================================================
# DSM-9  MODEL 3.0 — TRAINER
# src/model_3/train_v3.py
# ============================================================

import os
import json
import numpy as np
import torch
import torch.nn as nn
import xgboost as xgb
from datetime import datetime
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from safetensors.torch import save_file

from src.model_3.features_v3 import compute_features_v3, FEATURES_V3
from src.model_3.model_v3_0 import DSM9_v3, make_sequences, DEVICE, WINDOW, HORIZONS

# ── Training config ──────────────────────────────────────────
EPOCHS   = 400
PATIENCE = 30
LR       = 2e-4
SEED     = 42
MIN_ROWS = 500
MODEL_BASE = Path("model_storage/model_3")

# Quality gate — new model must beat old RMSE by this margin
QUALITY_GATE = 0.001

torch.manual_seed(SEED)
np.random.seed(SEED)


# ── Helpers ──────────────────────────────────────────────────

def load_old_rmse(ticker: str) -> float:
    p = MODEL_BASE / ticker / "metadata.json"
    if not p.exists():
        return float("inf")
    with open(p) as f:
        return float(json.load(f).get("val_rmse", float("inf")))


def needs_retrain(ticker: str, threshold_days: int = 14) -> bool:
    p = MODEL_BASE / ticker / "metadata.json"
    if not p.exists():
        return True
    with open(p) as f:
        meta = json.load(f)
    last = meta.get("last_retrained")
    if not last:
        return True
    from datetime import date
    days = (date.today() - datetime.fromisoformat(last).date()).days
    if days < threshold_days:
        print(f"  ⏩ {ticker} trained {days}d ago — skip")
        return False
    return True


# ── Main trainer ─────────────────────────────────────────────

def train_v3(ticker: str) -> dict:
    """
    Full Model 3.0 training pipeline for one ticker.
    Returns a status dict.
    """
    save_dir = MODEL_BASE / ticker
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── Features ────────────────────────────────────────────
    df = compute_features_v3(ticker)
    if len(df) < MIN_ROWS:
        return {"ticker": ticker, "status": "SKIPPED", "reason": f"Only {len(df)} rows"}

    for h in HORIZONS:
        df[f"target_{h}"] = df["Close"].shift(-h) / df["Close"] - 1
    df = df.dropna()

    X = df[FEATURES_V3].values
    y = df[[f"target_{h}" for h in HORIZONS]].values

    # ── Splits ──────────────────────────────────────────────
    t1 = int(len(X) * 0.70)
    t2 = int(len(X) * 0.85)
    X_train, y_train = X[:t1],    y[:t1]
    X_val,   y_val   = X[t1:t2],  y[t1:t2]

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)

    # ── XGBoost ─────────────────────────────────────────────
    print(f"  🌲 Training XGBoost...")
    xgb_models, xgb_val_preds = [], []

    for i, h in enumerate(HORIZONS):
        m = xgb.XGBRegressor(
            n_estimators=300, max_depth=6,
            learning_rate=0.02, subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist", random_state=SEED,
        )
        m.fit(X_train, y_train[:, i],
              eval_set=[(X_val, y_val[:, i])],
              verbose=False)
        xgb_models.append(m)
        xgb_val_preds.append(m.predict(X_val))

    xgb_val = np.column_stack(xgb_val_preds)    # (val_len, n_horizons)

    # ── Sequences for deep model ─────────────────────────────
    Xs_train, ys_train = make_sequences(X_train, y_train, WINDOW)
    Xs_val,   ys_val   = make_sequences(X_val,   y_val,   WINDOW)

    if len(Xs_train) == 0:
        return {"ticker": ticker, "status": "SKIPPED", "reason": "Not enough rows for WINDOW"}

    # XGB preds aligned to sequence length
    xgb_seq_val = xgb_val[WINDOW:]

    # ── DSM9_v3 deep model ──────────────────────────────────
    print(f"  🧠 Training DSM9_v3 (Transformer + BiLSTM)...")
    model = DSM9_v3(n_features=len(FEATURES_V3)).to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    lf    = nn.HuberLoss()

    # Pre-compute tensors
    Xt  = torch.tensor(Xs_train,  dtype=torch.float32).to(DEVICE)
    yt  = torch.tensor(ys_train,  dtype=torch.float32).to(DEVICE)
    Xv  = torch.tensor(Xs_val,    dtype=torch.float32).to(DEVICE)
    yv  = torch.tensor(ys_val,    dtype=torch.float32).to(DEVICE)
    xv_t = torch.tensor(xgb_seq_val, dtype=torch.float32).to(DEVICE)

    # XGB preds for train sequences
    xgb_seq_train = xgb_val[:len(Xs_train)]      # rough alignment (train split)
    xgb_tr_full   = np.column_stack([
        xgb_models[i].predict(X_train) for i in range(len(HORIZONS))
    ])
    xgb_tr_seq = xgb_tr_full[WINDOW:][:len(Xs_train)]
    xt_t = torch.tensor(xgb_tr_seq, dtype=torch.float32).to(DEVICE)

    best_val_loss = float("inf")
    best_weights  = None
    patience_ctr  = 0

    for epoch in range(EPOCHS):
        model.train(); opt.zero_grad()
        pred = model(Xt, xt_t)
        loss = lf(pred, yt)
        loss.backward(); opt.step(); sched.step()

        model.eval()
        with torch.no_grad():
            vp   = model(Xv, xv_t)
            vloss = lf(vp, yv).item()

        if vloss < best_val_loss:
            best_val_loss = vloss
            best_weights  = {k: v.clone() for k, v in model.state_dict().items()}
            patience_ctr  = 0
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f"    Early stop @ epoch {epoch}")
                break

    model.load_state_dict(best_weights)

    # ── Fused RMSE ──────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        deep_preds = model(Xv, xv_t).cpu().numpy()   # (val_seq_len, n_horizons)

    # Simple average fusion (deep model already learned internal fusion)
    fused_rmse = float(np.sqrt(mean_squared_error(ys_val, deep_preds)))
    old_rmse   = load_old_rmse(ticker)
    print(f"  📊 RMSE  new={fused_rmse:.5f}  old={old_rmse:.5f}")

    # ── Quality gate ────────────────────────────────────────
    if fused_rmse >= old_rmse - QUALITY_GATE and old_rmse != float("inf"):
        print("  🚫 Quality gate FAILED — keeping old model")
        return {
            "ticker": ticker, "status": "REJECTED",
            "reason": f"RMSE {fused_rmse:.5f} not better than {old_rmse:.5f}",
        }

    # ── Save ────────────────────────────────────────────────
    for i, h in enumerate(HORIZONS):
        xgb_models[i].save_model(str(save_dir / f"xgb_{h}.json"))

    save_file(model.state_dict(), str(save_dir / "model_v3.safetensors"))

    np.save(str(save_dir / "scaler_mean.npy"),  scaler.mean_)
    np.save(str(save_dir / "scaler_scale.npy"), scaler.scale_)

    metadata = {
        "version":        "3.0",
        "ticker":         ticker,
        "last_retrained": datetime.utcnow().isoformat(),
        "val_rmse":       fused_rmse,
        "horizons":       HORIZONS,
        "features":       FEATURES_V3,
        "n_features":     len(FEATURES_V3),
        "window":         WINDOW,
        "architecture":   "Transformer + BiLSTM_Attention + LearnedFusion + XGBoost",
        "confidence_score": round(max(0.3, 1 - fused_rmse * 10), 2),
        "training_rows":  len(df),
    }

    with open(save_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"  ✅ Model 3.0 saved → {save_dir}")
    return {"ticker": ticker, "status": "SUCCESS", "rmse": round(fused_rmse, 5)}
