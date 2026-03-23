from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

MODEL_DIR  = Path("model_storage/model_1")
TOP50_FILE = Path("data") / "nifty_top50.json"

# MongoDB
MONGODB_URI = os.getenv("MONGODB_URI")
MONGODB_DB  = os.getenv("MONGODB_DB", "dsm9")

if not MONGODB_URI:
    raise EnvironmentError("MONGODB_URI is not set. Add it to your .env file.")
