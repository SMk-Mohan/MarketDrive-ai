"""
api/routes/health.py
GET /health       → basic liveness check
GET /api-status   → full system status (models, cache, scheduler, APIs)
"""

import logging
import os
from datetime import datetime
from fastapi import APIRouter

from core.cache import get_all_predictions
from core.scheduler import scheduler
from agents.rag_agent import rag_agent
from config.settings import (
    STOCK_CONFIG,
    MODEL_DIR,
    VERSION,
    PROJECT_NAME,
)

logger = logging.getLogger(__name__)
router = APIRouter()


# ── GET /health ───────────────────────────────────────────────
@router.get("/health")
async def health():
    """Basic liveness check — used by deployment health probes."""
    return {
        "status":    "ok",
        "project":   PROJECT_NAME,
        "version":   VERSION,
        "timestamp": datetime.now().isoformat(),
    }


# ── GET /api-status ───────────────────────────────────────────
@router.get("/api-status")
async def api_status():
    """
    Full system status check:
    - Models loaded per stock
    - Predictions cached
    - Scheduler running
    - API budgets remaining
    """
    # 1. Model files
    models = {}
    for company in STOCK_CONFIG.keys():
        model_path = os.path.join(MODEL_DIR, f"{company}_model.pkl")
        models[company] = os.path.exists(model_path)

    # 2. Cache
    cached       = get_all_predictions()
    cache_status = {
        company: company in cached
        for company in STOCK_CONFIG.keys()
    }

    # 3. Scheduler
    scheduler_running = scheduler.running
    jobs = [
        {"id": job.id, "next_run": str(job.next_run_time)}
        for job in scheduler.get_jobs()
    ] if scheduler_running else []

    # 4. API budgets
    try:
        api_result = rag_agent("api_status", {})
        api_data   = api_result.get("data", {})
    except Exception:
        api_data = {}

    all_models_ready = all(models.values())
    system_ready     = all_models_ready and scheduler_running

    return {
        "status":           "ok" if system_ready else "degraded",
        "project":          PROJECT_NAME,
        "version":          VERSION,
        "timestamp":        datetime.now().isoformat(),
        "system_ready":     system_ready,
        "models": {
            "all_loaded":   all_models_ready,
            "per_stock":    models,
        },
        "cache": {
            "per_stock":    cache_status,
            "total_cached": sum(cache_status.values()),
        },
        "scheduler": {
            "running":      scheduler_running,
            "jobs":         jobs,
        },
        "api_budgets":      api_data,
    }