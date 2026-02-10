# ================================
# TRAIN ALL TOP-50 MODELS
# ================================

from pathlib import Path
from universe import load_top50
from model_1_0 import train_legendary_hybrid

BASE_DIR = Path("DSM-9")
MODEL_STORAGE = BASE_DIR / "model_storage"
MODEL_STORAGE.mkdir(exist_ok=True)


def train_all_top50(skip_existing=True):
    tickers = load_top50()
    print(f"\n🚀 Training started for {len(tickers)} companies\n")

    for i, ticker in enumerate(tickers, 1):
        print("=" * 60)
        print(f"[{i}/{len(tickers)}] {ticker}")

        out_dir = MODEL_STORAGE / ticker

        if skip_existing and (out_dir / "meta.json").exists():
            print("⏩ Model already exists — skipping")
            continue

        try:
            train_legendary_hybrid(
                ticker=ticker,
                output_dir=str(out_dir)
            )
            print(f"✅ Saved → {out_dir}")

        except Exception as e:
            print(f"❌ Failed for {ticker}")
            print(e)

    print("\n🎉 ALL TRAINING DONE")


if __name__ == "__main__":
    train_all_top50()
