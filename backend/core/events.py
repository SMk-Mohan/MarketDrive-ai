import logging
from core.scheduler import start_scheduler, stop_scheduler
from core.cache import init_cache

logger = logging.getLogger(__name__)


async def startup():
    logger.info("MarketDrive AI starting up...")
    from core.db_sync import download_all
    download_all()  # Restore state from cloud DB
    init_cache()
    start_scheduler()
    logger.info("Scheduler started. Cache initialised.")


async def shutdown():
    logger.info("MarketDrive AI shutting down...")
    stop_scheduler()
    logger.info("Scheduler stopped.")