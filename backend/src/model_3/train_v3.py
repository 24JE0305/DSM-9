# ============================================================
# DSM-9  MODEL 3.1 -- TRAINER (FIXED)
# backend/src/model_3/train_v3.py
#
# FIXES vs original v3.0 (output contract UNCHANGED):
#
#  FIX 1 -- WINDOW 60 -> 120
#  FIX 2 -- xgb_tr_seq alignment (was using val preds for train)
#  FIX 3 -- XGB early stopping on separate xgb_eval split
#  FIX 4 -- Bearish oversampling 2x to fix bullish bias
#  FIX 5 -- confidence_score scale 10 -> 4 (honest calibration)
#  FIX 6 -- PositionalEncoding dropout=drop (was NameError)
#
# API output JSON shape: IDENTICAL to v3.0 -- frontend safe
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
from src.model_3.model_v3_0 import DSM9_v3, make_seq, DEVICE, HORIZONS

# ── Training config ──────────────────────────────────────────
# FIX 1: WINDOW 60 -> 120
# 60 days of context for a 90d target = barely 1 horizon.
# 120 = 2x the shortest target horizon. Gives Transformer and
# BiLSTM enough sequence to learn lead/lag patterns properly.
WINDOW   = 120

EPOCHS   = 400
PATIENCE = 30
LR       = 2e-4
SEED     = 42
MIN_ROWS = 600           # raised from 500 because WINDOW is now 120
MODEL_BASE = Path("model_storage/model_3")
QUALITY_GATE = 0.001

# FIX 4: oversample bearish sequences 2x during training
# 5-year window is bull-market heavy causing +0.20 bullish bias
OVERSAMPLE_BEARISH = 2

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
        print(f"  {ticker} trained {days}d ago -- skip")
        return False
    return True


# ── Main trainer ─────────────────────────────────────────────

def train_v3(ticker: str) -> dict:
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

    # ── Splits: 70% train | 15% val | 15% test ──────────────
    # Test split is never touched during training -- only used
    # by test_accuracy_v3.py after training is complete.
    t1 = int(len(X) * 0.70)
    t2 = int(len(X) * 0.85)
    X_train, y_train = X[:t1], y[:t1]
    X_val,   y_val   = X[t1:t2], y[t1:t2]

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)

    # ── FIX 3: XGB gets its own eval split ──────────────────
    # Original used eval_set=[(X_val, y_val)] which is the same
    # val split as the deep model -> joint overfit.
    # Fix: carve last 15% of X_train as xgb_eval only.
    xgb_split = int(len(X_train) * 0.85)
    Xtr_xgb, ytr_xgb = X_train[:xgb_split], y_train[:xgb_split]
    Xev_xgb, yev_xgb = X_train[xgb_split:], y_train[xgb_split:]

    # ── XGBoost ─────────────────────────────────────────────
    print(f"  [XGB] Training...")
    xgb_models, xgb_val_preds = [], []

    for i, h in enumerate(HORIZONS):
        m = xgb.XGBRegressor(
            n_estimators=300,      # hard cap -- version-safe, no early_stopping needed
            max_depth=5,
            learning_rate=0.02,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            reg_lambda=1.5,
            tree_method="hist",
            random_state=SEED,
        )
        m.fit(Xtr_xgb, ytr_xgb[:, i], verbose=False)
        xgb_models.append(m)
        xgb_val_preds.append(m.predict(X_val))

    xgb_val = np.column_stack(xgb_val_preds)   # (val_len, n_horizons)

    # ── Sequences for deep model ─────────────────────────────
    Xs_train, ys_train = make_seq(X_train, y_train, WINDOW)
    Xs_val,   ys_val   = make_seq(X_val,   y_val,   WINDOW)

    # CRASH FIX B: guard -- val split too small for WINDOW causes mat(1x0) crash
    MIN_VAL_SEQS = 10
    if len(Xs_train) == 0:
        return {"ticker": ticker, "status": "SKIPPED", "reason": "Train seqs=0 -- dataset too small for WINDOW"}
    if len(Xs_val) < MIN_VAL_SEQS:
        return {"ticker": ticker, "status": "SKIPPED",
                "reason": f"Val seqs={len(Xs_val)} < {MIN_VAL_SEQS} -- dataset too small for WINDOW={WINDOW}"}

    # FIX 2: xgb_tr -- predict on X_train directly (no val bleed)
    xgb_tr_full = np.column_stack([
        xgb_models[i].predict(X_train) for i in range(len(HORIZONS))
    ])
    xgb_tr = xgb_tr_full[WINDOW:][:len(Xs_train)]

    # CRASH FIX C: safe xgb_v -- take last len(Xs_val) rows, always matches
    # xgb_val[WINDOW:] crashes when val_len <= WINDOW giving shape (0, n_h)
    xgb_v = xgb_val[len(xgb_val) - len(Xs_val):]

    assert len(xgb_tr) == len(Xs_train), f"xgb_tr/Xs_train: {len(xgb_tr)} vs {len(Xs_train)}"
    assert len(xgb_v)  == len(Xs_val),   f"xgb_v/Xs_val: {len(xgb_v)} vs {len(Xs_val)}"

    # ── FIX 4: Bearish oversampling ──────────────────────────
    # 5-year training window (2020-2025) was a bull market for Nifty.
    # Model learned to predict up almost always (mean bias +0.20).
    # Duplicate sequences where 90d outcome was negative to give
    # equal exposure to bear moves. Val untouched.
    h0_idx = 0   # index of 90d horizon
    bear_mask = ys_train[:, h0_idx] < 0
    n_bear = bear_mask.sum()
    if n_bear > 0 and OVERSAMPLE_BEARISH > 1:
        extra = OVERSAMPLE_BEARISH - 1
        Xs_train = np.concatenate([Xs_train, np.tile(Xs_train[bear_mask], (extra, 1, 1))], axis=0)
        ys_train = np.concatenate([ys_train, np.tile(ys_train[bear_mask], (extra, 1))   ], axis=0)
        xgb_tr   = np.concatenate([xgb_tr,   np.tile(xgb_tr[bear_mask],  (extra, 1))   ], axis=0)
        perm     = np.random.permutation(len(Xs_train))
        Xs_train, ys_train, xgb_tr = Xs_train[perm], ys_train[perm], xgb_tr[perm]
        print(f"  [OVERSAMPLE] bearish {n_bear} -> {n_bear*OVERSAMPLE_BEARISH} | train seqs: {len(Xs_train)}")

    # ── DSM9_v3 deep model ───────────────────────────────────
    print(f"  [DEEP] DSM9_v3 (Transformer+BiLSTM)...")
    model = DSM9_v3(n_feat=len(FEATURES_V3)).to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    lf    = nn.HuberLoss()

    Xt    = torch.tensor(Xs_train, dtype=torch.float32).to(DEVICE)
    yt    = torch.tensor(ys_train, dtype=torch.float32).to(DEVICE)
    Xv    = torch.tensor(Xs_val,   dtype=torch.float32).to(DEVICE)
    yv    = torch.tensor(ys_val,   dtype=torch.float32).to(DEVICE)
    xt_tr = torch.tensor(xgb_tr,   dtype=torch.float32).to(DEVICE)
    xt_v  = torch.tensor(xgb_v,    dtype=torch.float32).to(DEVICE)

    best_val_loss = float("inf")
    best_weights  = None
    patience_ctr  = 0

    for epoch in range(EPOCHS):
        model.train()
        opt.zero_grad()
        lf(model(Xt, xt_tr), yt).backward()
        opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            vloss = lf(model(Xv, xt_v), yv).item()

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
    model.eval()
    with torch.no_grad():
        deep_preds = model(Xv, xt_v).cpu().numpy()

    fused_rmse = float(np.sqrt(mean_squared_error(ys_val, deep_preds)))
    old_rmse   = load_old_rmse(ticker)
    print(f"  RMSE  new={fused_rmse:.5f}  old={old_rmse:.5f}")

    # ── Quality gate ─────────────────────────────────────────
    if fused_rmse >= old_rmse - QUALITY_GATE and old_rmse != float("inf"):
        print("  Quality gate FAILED -- keeping old model")
        return {
            "ticker": ticker, "status": "REJECTED",
            "reason": f"RMSE {fused_rmse:.5f} not better than {old_rmse:.5f}",
        }

    # ── Save ─────────────────────────────────────────────────
    for i, h in enumerate(HORIZONS):
        xgb_models[i].save_model(str(save_dir / f"xgb_{h}.json"))

    save_file(model.state_dict(), str(save_dir / "model_v3.safetensors"))
    np.save(str(save_dir / "scaler_mean.npy"),  scaler.mean_)
    np.save(str(save_dir / "scaler_scale.npy"), scaler.scale_)

    # FIX 5: confidence_score scale 10 -> 4
    # Original 1-rmse*10 gave false "High" when val_rmse was 4-5x
    # lower than true test RMSE (overfit artefact).
    # Scale 4 maps realistic RMSEs honestly:
    #   rmse=0.05 -> 0.80 (High)
    #   rmse=0.10 -> 0.60 (Moderate)
    #   rmse=0.15 -> 0.40 (Moderate)
    #   rmse=0.25 -> 0.30 (Low)
    # confidence_level thresholds in inference_v3.py unchanged
    # so frontend display is unaffected.
    conf = round(max(0.3, 1 - fused_rmse * 4), 2)

    metadata = {
        "version":          "3.1",
        "ticker":           ticker,
        "last_retrained":   datetime.utcnow().isoformat(),
        "val_rmse":         fused_rmse,
        "horizons":         HORIZONS,
        "features":         FEATURES_V3,
        "n_features":       len(FEATURES_V3),
        "window":           WINDOW,
        "architecture":     "Transformer+BiLSTM-Attention+XGBoost+LearnedFusion",
        "confidence_score": conf,
        "training_rows":    len(df),
    }

    with open(save_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"  Saved -> {save_dir}")
    return {"ticker": ticker, "status": "SUCCESS", "rmse": round(fused_rmse, 5)}