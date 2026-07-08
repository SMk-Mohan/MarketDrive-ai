"""
api/routes/prediction.py
GET /predict/{ticker}     → run full pipeline for one stock
GET /predict/all          → latest cached predictions for all stocks
POST /predict/run-predictions → trigger full background sweep for CI
"""

import logging
from fastapi import APIRouter, HTTPException, BackgroundTasks

from core.cache import get_prediction, get_all_predictions, set_prediction
from agents.coordinator import run_pipeline
from config.settings import STOCK_CONFIG

logger = logging.getLogger(__name__)
router = APIRouter()

VALID_TICKERS = list(STOCK_CONFIG.keys())  # ["infosys", "vodafone", ...]


def _resolve(ticker: str) -> str:
    """Normalise ticker → company name key used internally."""
    t = ticker.lower().strip()
    if t not in VALID_TICKERS:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown ticker '{ticker}'. Valid: {VALID_TICKERS}"
        )
    return t


async def background_sweep():
    """Runs all predictions concurrently and updates cache as they finish."""
    from agents.coordinator import run_pipeline
    import asyncio
    
    logger.info("🚀 [BackgroundSweep] Starting PARALLEL agent sweep...")
    
    async def process_one(company):
        try:
            logger.info(f"⏳ [BackgroundSweep] Starting {company}...")
            # 120s timeout per stock
            result = await asyncio.wait_for(run_pipeline(company), timeout=120.0)
            
            if result and "prediction" in result:
                set_prediction(company, result)
                logger.info(f"✅ [BackgroundSweep] {company} completed.")
            else:
                logger.error(f"❌ [BackgroundSweep] {company} returned invalid data.")
        except asyncio.TimeoutError:
            logger.error(f"⏰ [BackgroundSweep] TIMEOUT for {company}")
        except Exception as e:
            logger.error(f"❌ [BackgroundSweep] {company} error: {e}")

    # Start all 5 stocks at the same time
    await asyncio.gather(*(process_one(c) for c in VALID_TICKERS))
    logger.info("🏁 [BackgroundSweep] All agents finished.")

# ── GET /predict/all ──────────────────────────────────────────
@router.get("/all")
async def get_all(background_tasks: BackgroundTasks, refresh: bool = False):
    """
    Returns latest cached predictions. 
    """
    data = get_all_predictions()
    
    # Trigger sweep if missing stocks or refresh requested
    if refresh or len(data) < len(VALID_TICKERS):
        logger.info(f"📡 [/predict/all] Triggering PARALLEL sweep. Current count: {len(data)}")
        background_tasks.add_task(background_sweep)

    return {
        "status":  "ok",
        "count":   len(data),
        "data":    data,
        "is_refreshing": refresh or len(data) < len(VALID_TICKERS)
    }


# ── POST /predict/run-predictions ─────────────────────────────
@router.post("/run-predictions")
async def run_predictions_ci(background_tasks: BackgroundTasks):
    """
    Dedicated endpoint for GitHub Actions to trigger the daily sweep.
    Returns immediately and runs logic in background.
    """
    logger.info("📡 [CI Trigger] Manual sweep requested via /run-predictions")
    background_tasks.add_task(background_sweep)
    return {
        "status":  "accepted",
        "message": "Background prediction sweep started for all stocks.",
        "stocks":  VALID_TICKERS
    }



# ── GET /predict/{ticker} ─────────────────────────────────────
@router.get("/{ticker}")
async def get_prediction_for_ticker(ticker: str):
    """
    Returns cached prediction for one stock.
    If cache is empty (first run before 9:15 AM), triggers pipeline live.
    """
    company = _resolve(ticker)
    cached  = get_prediction(company)

    if cached:
        return {"status": "ok", "source": "cache", "data": cached}

    # Cache miss — run pipeline live (first request of the day)
    logger.info(f"[/predict/{ticker}] Cache miss — running pipeline live")
    try:
        result = await run_pipeline(company)
        set_prediction(company, result)
        return {"status": "ok", "source": "live", "data": result}
    except Exception as e:
        logger.error(f"[/predict/{ticker}] Pipeline error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── POST /predict/{ticker}/force ──────────────────────────────
@router.post("/{ticker}/force")
async def force_predict(ticker: str, background_tasks: BackgroundTasks):
    """
    Force re-runs the full pipeline for one stock and updates cache.
    Used for manual trigger or testing. Runs in background.
    """
    company = _resolve(ticker)

    async def _run():
        try:
            result = await run_pipeline(company)
            set_prediction(company, result)
            logger.info(f"[/predict/{ticker}/force] Done: {result.get('prediction')}")
        except Exception as e:
            logger.error(f"[/predict/{ticker}/force] Error: {e}")

    background_tasks.add_task(_run)
    return {
        "status":  "accepted",
        "message": f"Force re-prediction started for {ticker}",
    }