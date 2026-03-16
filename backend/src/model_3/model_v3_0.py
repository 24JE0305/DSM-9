# ============================================================
# DSM-9  MODEL 3.0 — ARCHITECTURE
# src/model_3/model_v3_0.py
#
# Exact mirror of notebook Cell 6.
# FIX: PositionalEncoding(d, dropout=dropout) → dropout=drop
#      ('dropout' was undefined in __init__ scope — NameError)
# ============================================================

import torch
import torch.nn as nn
import numpy as np

DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
WINDOW   = 60
HORIZONS = [90, 365]


class PositionalEncoding(nn.Module):
    def __init__(self, d, max_len=512, dropout=0.1):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        pe  = torch.zeros(max_len, d)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2).float() * (-np.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.drop(x + self.pe[:, :x.size(1)])


class TransformerBlock(nn.Module):
    def __init__(self, d, heads=4, ff=256, drop=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, heads, dropout=drop, batch_first=True)
        self.n1   = nn.LayerNorm(d)
        self.n2   = nn.LayerNorm(d)
        self.ff   = nn.Sequential(
            nn.Linear(d, ff), nn.GELU(), nn.Dropout(drop), nn.Linear(ff, d)
        )
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        a, _ = self.attn(x, x, x)
        x    = self.n1(x + self.drop(a))
        return self.n2(x + self.drop(self.ff(x)))


class TemporalAttention(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.a = nn.Linear(h, 1)

    def forward(self, x):
        w = torch.softmax(self.a(x), dim=1)
        return (w * x).sum(dim=1), w


class BiLSTMAttn(nn.Module):
    def __init__(self, n, h=128, layers=2, drop=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            n, h, layers,
            batch_first=True,
            bidirectional=True,
            dropout=drop if layers > 1 else 0.0,
        )
        self.attn = TemporalAttention(h * 2)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        o, _ = self.lstm(x)
        c, w = self.attn(o)
        return self.drop(c), w


class DSM9_v3(nn.Module):
    def __init__(self, n_feat, d=64, heads=4, n_trans=2,
                 lstm_h=128, lstm_l=2, n_h=2, drop=0.2):
        super().__init__()
        self.proj        = nn.Linear(n_feat, d)
        self.pe          = PositionalEncoding(d, dropout=drop)   # FIX: was dropout=dropout (NameError)
        self.transformer = nn.Sequential(
            *[TransformerBlock(d, heads, d * 4, drop) for _ in range(n_trans)]
        )
        self.bilstm = BiLSTMAttn(n_feat, lstm_h, lstm_l, drop)
        self.fusion = nn.Sequential(
            nn.Linear(d + lstm_h * 2 + n_h, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, n_h),
        )
        self.n_h = n_h

    def forward(self, x, xgb=None):
        B  = x.size(0)
        t  = self.pe(self.proj(x))
        for layer in self.transformer:
            t = layer(t)
        tf      = t.mean(dim=1)
        lf, _   = self.bilstm(x)
        if xgb is None:
            xgb = torch.zeros(B, self.n_h, device=x.device)
        return self.fusion(torch.cat([tf, lf, xgb], dim=-1))


def make_seq(X, y, w):
    return (
        np.array([X[i:i + w] for i in range(len(X) - w)]),
        np.array([y[i + w]   for i in range(len(X) - w)]),
    )