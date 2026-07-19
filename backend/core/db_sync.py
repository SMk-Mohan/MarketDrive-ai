import os
import base64
import logging
from pathlib import Path
from pymongo import MongoClient

logger = logging.getLogger("db_sync")

MONGODB_URI = os.getenv("MONGODB_URI", "")

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
CACHE_DIR = BASE_DIR / "cache"
RAG_DIR = BASE_DIR / "rag"

# List of files we specifically want to sync
FILES_TO_SYNC = [
    # Cache
    CACHE_DIR / "prediction_cache.json",
    CACHE_DIR / "market_cache.json",
    # RAG reports
    RAG_DIR / "market_time_report.csv",
    RAG_DIR / "global_report.csv",
    RAG_DIR / "api_tracker.json",
    RAG_DIR / "training_data_tracker.csv",
    # FAISS indexes (10 files: 5 stocks * 2 files)
    RAG_DIR / "faiss_indexes/infosys_faiss.index",
    RAG_DIR / "faiss_indexes/infosys_rows.pkl",
    RAG_DIR / "faiss_indexes/vodafone_faiss.index",
    RAG_DIR / "faiss_indexes/vodafone_rows.pkl",
    RAG_DIR / "faiss_indexes/tata_faiss.index",
    RAG_DIR / "faiss_indexes/tata_rows.pkl",
    RAG_DIR / "faiss_indexes/adani_faiss.index",
    RAG_DIR / "faiss_indexes/adani_rows.pkl",
    RAG_DIR / "faiss_indexes/yesbank_faiss.index",
    RAG_DIR / "faiss_indexes/yesbank_rows.pkl",
]

_mongo_client = None

def get_mongo_collection():
    global _mongo_client
    if not MONGODB_URI:
        return None
    try:
        if _mongo_client is None:
            # pymongo MongoClient handles connection pooling
            _mongo_client = MongoClient(MONGODB_URI)
        db = _mongo_client["marketdrive"]
        return db["app_state"]
    except Exception as e:
        logger.error(f"[DB Sync] MongoDB connection error: {e}")
        return None

def is_db_configured() -> bool:
    return bool(MONGODB_URI)

def upload_file(filepath: str | Path):
    if not is_db_configured():
        return
    
    col = get_mongo_collection()
    if col is None:
        return

    path = Path(filepath)
    if not path.exists():
        logger.warning(f"[DB Sync] File does not exist locally to upload: {path}")
        return

    try:
        rel_key = str(path.relative_to(BASE_DIR)).replace("\\", "/")
        is_binary = path.suffix in ['.index', '.pkl']
        
        if is_binary:
            with open(path, "rb") as f:
                content = base64.b64encode(f.read()).decode('utf-8')
        else:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()

        # MongoDB upsert using _id as key
        col.replace_one(
            {"_id": rel_key},
            {
                "content": content,
                "is_binary": is_binary
            },
            upsert=True
        )
        logger.info(f"[DB Sync] Successfully uploaded {rel_key} to MongoDB.")
    except Exception as e:
        logger.error(f"[DB Sync] Error uploading {filepath}: {e}")

def download_file(rel_key: str):
    if not is_db_configured():
        return

    col = get_mongo_collection()
    if col is None:
        return

    try:
        doc = col.find_one({"_id": rel_key})
        if not doc:
            logger.info(f"[DB Sync] No cloud file found for key in MongoDB: {rel_key}")
            return

        content = doc.get("content", "")
        is_binary = doc.get("is_binary", False)
        dest_path = BASE_DIR / rel_key

        dest_path.parent.mkdir(parents=True, exist_ok=True)

        if is_binary:
            with open(dest_path, "wb") as f:
                f.write(base64.b64decode(content))
        else:
            with open(dest_path, "w", encoding="utf-8") as f:
                f.write(content)
        
        logger.info(f"[DB Sync] Successfully restored {rel_key} from MongoDB.")
    except Exception as e:
        logger.error(f"[DB Sync] Error downloading {rel_key}: {e}")

def download_all():
    if not is_db_configured():
        logger.info("[DB Sync] MongoDB not configured. Using local filesystem only.")
        return

    logger.info("[DB Sync] Restoring application state from MongoDB...")
    for path in FILES_TO_SYNC:
        rel_key = str(path.relative_to(BASE_DIR)).replace("\\", "/")
        download_file(rel_key)

def upload_all():
    if not is_db_configured():
        return

    logger.info("[DB Sync] Backing up all application state to MongoDB...")
    for path in FILES_TO_SYNC:
        if path.exists():
            upload_file(path)
