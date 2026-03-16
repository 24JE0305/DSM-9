import yfinance as yf
import pandas as pd
from universe import load_top50
from pathlib import Path

DATA_DIR = Path("DSM-9/data_cache")
DATA_DIR.mkdir(exist_ok=True)

for ticker in load_top50():
    df = yf.download(ticker, period="6mo", progress=False)
    df.to_csv(DATA_DIR / f"{ticker}.csv")
    print(f"Updated {ticker}")
