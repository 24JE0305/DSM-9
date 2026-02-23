import os
import json
from datetime import datetime

from src.model_2.model_v2_0 import train_strong_hybrid


DATA_FILE = "data/nifty_top50.json"
BASE_SAVE_DIR = "model_storage/model_2"


def train_symbol(symbol: str):

    print("\n" + "=" * 70)
    print(f"🚀 Training Strong Hybrid Model 2.0 for {symbol}")
    print("=" * 70)

    start_time = datetime.now()

    symbol_save_dir = os.path.join(BASE_SAVE_DIR, symbol)
    os.makedirs(symbol_save_dir, exist_ok=True)

    try:
        metrics = train_strong_hybrid(
            ticker=symbol,
            save_dir=symbol_save_dir
        )

        print(f"📊 Metrics for {symbol}")
        print(metrics)

        print(f"✅ Finished {symbol} in {datetime.now() - start_time}")

    except Exception as e:
        print(f"❌ Failed training {symbol}")
        print(str(e))


def main():

    print("\n📊 MODEL_2 BATCH TRAINING STARTED")

    with open(DATA_FILE, "r") as f:
        symbols = json.load(f)

    # 🔥 IMPORTANT — START SMALL
    symbols = symbols[:5]

    for idx, symbol in enumerate(symbols, start=1):
        print(f"\n[{idx}/{len(symbols)}] Processing {symbol}")
        train_symbol(symbol)

    print("\n🎉 Batch training complete!")


if __name__ == "__main__":
    main()