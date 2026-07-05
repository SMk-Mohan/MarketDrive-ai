import logging
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from config import settings

logger    = logging.getLogger(__name__)
scheduler = AsyncIOScheduler(timezone="Asia/Kolkata")


# ── jobs ──────────────────────────────────────────────────────────────────────

async def run_daily_prediction():
    """9:15 AM IST — run full prediction pipeline for all stocks."""
    from agents.coordinator import run_pipeline
    logger.info("Scheduler: running daily prediction pipeline...")
    for ticker in settings.STOCKS:
        try:
            await run_pipeline(ticker)
            logger.info(f"Scheduler: prediction complete for {ticker}")
        except Exception as e:
            logger.error(f"Scheduler: prediction failed for {ticker}: {e}")


async def run_evaluation():
    """3:30 PM IST — evaluate today's predictions against actual close."""
    from agents.coordinator import run_evaluation
    logger.info("Scheduler: running end-of-day evaluation...")
    try:
        await run_evaluation()
    except Exception as e:
        logger.error(f"Scheduler: evaluation failed: {e}")


async def run_cleanup():
    """4:30 PM IST — clear stale market cache, update RAG memory."""
    from core.cache import clear_cache
    logger.info("Scheduler: running cleanup...")
    clear_cache()


async def run_retraining():
    """4:30 PM IST — trigger daily incremental retraining for all models."""
    from agents.coordinator import run_retraining
    logger.info("Scheduler: starting daily retraining process...")
    try:
        await run_retraining()
        logger.info("Scheduler: retraining complete for all models.")
    except Exception as e:
        logger.error(f"Scheduler: retraining failed: {e}")


# ── start / stop ──────────────────────────────────────────────────────────────

def start_scheduler():
    pred_h, pred_m   = settings.PREDICTION_TIME.split(":")
    eval_h, eval_m   = settings.EVALUATION_TIME.split(":")
    clean_h, clean_m = settings.CLEANUP_TIME.split(":")
    retrain_h, retrain_m = settings.RETRAIN_TIME.split(":")

    scheduler.add_job(
        run_daily_prediction,
        CronTrigger(hour=pred_h, minute=pred_m),
        id="daily_prediction",
        replace_existing=True
    )
    scheduler.add_job(
        run_evaluation,
        CronTrigger(hour=eval_h, minute=eval_m),
        id="evaluation",
        replace_existing=True
    )
    scheduler.add_job(
        run_cleanup,
        CronTrigger(hour=clean_h, minute=clean_m),
        id="cleanup",
        replace_existing=True
    )
    scheduler.add_job(
        run_retraining,
        CronTrigger(hour=retrain_h, minute=retrain_m),
        id="retraining",
        replace_existing=True
    )

    scheduler.start()
    logger.info(f"Scheduler started — prediction@{settings.PREDICTION_TIME}, eval@{settings.EVALUATION_TIME}, cleanup@{settings.CLEANUP_TIME} IST")


def stop_scheduler():
    scheduler.shutdown(wait=False)