# DSM-9 — AI-Powered Stock Market Forecasting Platform

DSM-9 is a full-stack market forecasting project for Indian equities. It combines:
- **FastAPI backend** for inference APIs,
- **React + Vite frontend** for visualization,
- **Hybrid ML models** (XGBoost + LSTM) for multi-horizon stock predictions,
- **Data cache + training pipelines** for continuous updates.

The project currently supports two model tracks:
- **Model 1.0**: classic hybrid forecasting pipeline,
- **Model 2.0**: stronger feature-engineered hybrid pipeline with richer output and risk signals.

---

## Project Structure

```text
DSM-9/
├── app/                         # FastAPI service (inference + API routes)
│   ├── main.py                  # API entrypoint
│   ├── inference.py             # Model 1.0 inference logic
│   ├── schemas.py               # Pydantic request/response models
│   └── model_2/
│       └── inference_v2.py      # Model 2.0 inference logic
├── src/                         # Training and data refresh scripts
│   ├── model_1_0.py             # Model 1.0 training pipeline
│   ├── train_all.py             # Batch training for top universe
│   ├── update_cache.py          # Refresh cached OHLCV data
│   └── model_2/                 # Model 2.0 training + utils
├── data/                        # Universe inputs (top50, nifty500)
├── data_cache/                  # Local per-ticker historical cache
├── model_storage/               # Saved trained models (generated)
├── frontend/                    # React dashboard
├── requirements.txt             # Python dependencies
└── README.md
```

---

## Features

- **Ticker universe endpoints** (`/tickers`) for frontend dropdown integration.
- **Historical time-series API** (`/history/{ticker}`) for chart rendering.
- **Model 1.0 prediction endpoint** (`POST /predict`) with multi-horizon forecast values.
- **Model 2.0 prediction endpoint** (`GET /predict_v2/{ticker}`) with:
  - expected return,
  - confidence metrics,
  - forecast range,
  - directional signal bias,
  - basic risk indicators.
- **Top-50 focused workflow** (based on `data/nifty_top50.json`).

---

## Tech Stack

### Backend
- Python
- FastAPI
- Pydantic
- Pandas / NumPy
- PyTorch
- XGBoost
- yfinance
- safetensors

### Frontend
- React
- Vite
- Tailwind CSS
- Axios
- Recharts
- Lucide icons

---

## Setup

## 1) Clone repository

```bash
git clone <your-repo-url>
cd DSM-9
```

## 2) Python environment

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows PowerShell

pip install -r requirements.txt
# Add API/runtime packages if missing:
pip install fastapi uvicorn xgboost safetensors scikit-learn
```

## 3) Frontend dependencies

```bash
cd frontend
npm install
cd ..
```

---

## Running the Platform

### Start backend (FastAPI)

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Backend base URL: `http://localhost:8000`

### Start frontend (React)

```bash
cd frontend
npm run dev
```

Frontend URL (default): `http://localhost:5173`

---

## API Reference

## Health

```http
GET /
```

Response:

```json
{ "status": "ok" }
```

## Universe

```http
GET /tickers
```

Returns top ticker list from `data/nifty_top50.json`.

## Historical Data

```http
GET /history/{ticker}
```

Returns cached historical rows for the ticker from `data_cache/{ticker}.csv`.

## Model 1.0 Prediction

```http
POST /predict
Content-Type: application/json

{
  "ticker": "RELIANCE.NS"
}
```

Typical response:

```json
{
  "ticker": "RELIANCE.NS",
  "last_close": 2900.25,
  "predictions": {
    "7D": 2935.61,
    "30D": 2992.42,
    "90D": 3120.17,
    "365D": 3351.80
  }
}
```

## Model 2.0 Prediction

```http
GET /predict_v2/RELIANCE.NS
```

Returns richer per-horizon objects with confidence and risk metadata.

---

## Training & Data Update Workflows

## Update local cache

```bash
python src/update_cache.py
```

Downloads latest history for universe symbols into `data_cache/`.

## Train Model 1.0 (batch)

```bash
python src/train_all.py
```

Default mode in script is update-oriented retraining. You can adjust behavior in the script (`fresh` vs `update`).

## Train Model 2.0 (batch)

```bash
python src/model_2/train_model_2_all.py
```

By default, this script currently trains a small subset first (`symbols[:5]`) for safer iteration.

---

## Notes / Caveats

- `requirements.txt` contains the core ML/data libraries, but backend serving packages may still need explicit install depending on your environment.
- Training is compute-intensive (especially LSTM) and can be slow on CPU.
- GPU is used when available in model training code paths.
- This project provides **research forecasts**, not investment advice.

---

## Quick Validation Checklist

After setup:

1. Start backend and open `http://localhost:8000`.
2. Verify `GET /tickers` returns symbols.
3. Start frontend and trigger **Analyze** for a ticker.
4. Confirm chart + forecast cards render without API errors.

---

## Disclaimer

This software is for educational and research purposes only. Market predictions are probabilistic and uncertain. Do not treat outputs as financial advice.
