# ================================
# TRAIN / UPDATE ALL TOP-50
# ================================

from pathlib import Path
from universe import load_top50
from model_1_0 import train_legendary_hybrid

BASE_DIR = Path("DSM-9")
MODEL_STORAGE = BASE_DIR / "model_storage"
MODEL_STORAGE.mkdir(exist_ok=True)


def train_all_top50(mode="fresh"):
    tickers = load_top50()
    print(f"\n🚀 MODE: {mode.upper()} | {len(tickers)} COMPANIES\n")

    for i, ticker in enumerate(tickers, 1):
        print("=" * 60)
        print(f"[{i}/{len(tickers)}] {ticker}")

        out_dir = MODEL_STORAGE / ticker

        try:
            train_legendary_hybrid(
                ticker=ticker,
                output_dir=str(out_dir),
                mode=mode
            )
        except Exception as e:
            print(f"❌ FAILED: {ticker}")
            print(e)

    print("\n🎉 ALL DONE")


if __name__ == "__main__":
    # fresh → first time
    # update → retrain if ≥ 14 days new data
    train_all_top50(mode="update")
