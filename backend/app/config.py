from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

MODEL_DIR  = Path("model_storage/model_1")
TOP50_FILE = Path("data") / "nifty_top50.json"

# MongoDB
MONGODB_URI = os.getenv(
    "MONGODB_URI",
    "MONGODB_URI_PLACEHOLDER"
)
MONGODB_DB  = os.getenv("MONGODB_DB", "dsm9")
