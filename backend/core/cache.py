import json
import os
import logging
from datetime import datetime
from config import settings

logger = logging.getLogger(__name__)

# In-memory store — holds latest prediction per ticker
_prediction_cache: dict = {}
_market_cache:     dict = {}

# ── init ──────────────────────────────────────────────────────────────────────

def init_cache():
    """Load persisted cache files from disk on startup."""
    global _prediction_cache, _market_cache

    pred_path   = os.path.join(settings.CACHE_DIR, "prediction_cache.json")
    market_path = os.path.join(settings.CACHE_DIR, "market_cache.json")

    if os.path.exists(pred_path):
        with open(pred_path) as f:
            _prediction_cache = json.load(f)
        logger.info(f"Prediction cache loaded: {list(_prediction_cache.keys())}")

    if os.path.exists(market_path):
        with open(market_path) as f:
            _market_cache = json.load(f)
        logger.info(f"Market cache loaded: {list(_market_cache.keys())}")


def _persist(path: str, data: dict):
    os.makedirs(settings.CACHE_DIR, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


# ── prediction cache ──────────────────────────────────────────────────────────

def set_prediction(ticker: str, data: dict):
    data["cached_at"] = datetime.utcnow().isoformat()
    _prediction_cache[ticker] = data
    _persist(os.path.join(settings.CACHE_DIR, "prediction_cache.json"), _prediction_cache)


def get_prediction(ticker: str) -> dict | None:
    return _prediction_cache.get(ticker)


def get_all_predictions() -> dict:
    return _prediction_cache


# ── market cache ──────────────────────────────────────────────────────────────

def set_market(ticker: str, data: dict):
    data["cached_at"] = datetime.utcnow().isoformat()
    _market_cache[ticker] = data
    _persist(os.path.join(settings.CACHE_DIR, "market_cache.json"), _market_cache)


def get_market(ticker: str) -> dict | None:
    return _market_cache.get(ticker)


def clear_cache():
    global _prediction_cache, _market_cache
    _prediction_cache = {}
    _market_cache     = {}
    logger.info("Cache cleared.")