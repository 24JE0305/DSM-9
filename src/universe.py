import json
import datetime
import time
import pandas as pd
import yfinance as yf
from pathlib import Path

# ==============================
# PATHS
# ==============================

DATA_DIR = Path("DSM-9/data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

UNIVERSE_FILE = DATA_DIR / "nifty500.csv"
CACHE_FILE = DATA_DIR / "market_caps_cache.json"
OUTPUT_FILE = DATA_DIR / "nifty_top50.json"

print("[INIT] Data directory:", DATA_DIR)

# ==============================
# LOAD NIFTY 500
# ==============================


def load_nifty500_universe():
    print("[STEP 1] Loading NIFTY 500 universe...")

    if not UNIVERSE_FILE.exists():
        raise FileNotFoundError("nifty500.csv not found")

    df = pd.read_csv(UNIVERSE_FILE)
    symbols = df["Symbol"].dropna().unique().tolist()

    tickers = [s + ".NS" for s in symbols]
    print(f"[OK] Loaded {len(tickers)} symbols from CSV")
    return tickers

# ==============================
# CACHE HANDLING
# ==============================


def load_cache():
    if CACHE_FILE.exists():
        with open(CACHE_FILE) as f:
            cache = json.load(f)
        print(f"[CACHE] Loaded {len(cache)} cached market caps")
        return cache
    return {}


def save_cache(cache):
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)

# ==============================
# FETCH MARKET CAP (SAFE)
# ==============================


def fetch_market_cap(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.get_info()   # safer than .info
        return info.get("marketCap")
    except Exception as e:
        print(f"[WARN] Failed for {ticker}: {e}")
        return None

# ==============================
# GENERATE TOP 50
# ==============================


def generate_top50():
    print("\n[STEP 2] Starting Top-50 generation")

    tickers = load_nifty500_universe()
    cache = load_cache()

    total = len(tickers)
    updated = False

    print(f"[INFO] Fetching market caps for {total} companies")

    for idx, ticker in enumerate(tickers, 1):
        if ticker in cache:
            continue

        print(f"[FETCH] {idx}/{total} | {ticker}")

        mc = fetch_market_cap(ticker)
        if mc:
            cache[ticker] = mc
            updated = True
            print(f"       Market Cap OK")
        else:
            print(f"       Market Cap MISSING")

        # ---- SAVE PROGRESS EVERY 5 REQUESTS ----
        if idx % 5 == 0:
            save_cache(cache)
            print("[SAVE] Progress saved to cache")

        time.sleep(1.5)  # RATE LIMIT PROTECTION

    if updated:
        save_cache(cache)
        print("[SAVE] Final cache saved")

    if len(cache) < 50:
        raise RuntimeError("Not enough valid market cap data")

    df = (
        pd.DataFrame(cache.items(), columns=["ticker", "market_cap"])
        .sort_values("market_cap", ascending=False)
        .head(50)
    )

    output = {
        "generated_on": datetime.date.today().isoformat(),
        "count": len(df),
        "tickers": df["ticker"].tolist()
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)

    print("[DONE] Top-50 stored in nifty_top50.json")
    return output

# ==============================
# LOAD STORED TOP 50
# ==============================


def load_top50():
    if not OUTPUT_FILE.exists():
        raise FileNotFoundError(
            "Top-50 not found. Run generate_top50() first.")

    with open(OUTPUT_FILE) as f:
        return json.load(f)["tickers"]

# ==============================
# RUN
# ==============================


if __name__ == "__main__":
    result = generate_top50()
    print("\n[RESULT]")
    print(result)
