from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from core.events import startup, shutdown
from api.routes import prediction, market, news, evaluation, report, health


@asynccontextmanager
async def lifespan(app: FastAPI):
    await startup()
    yield
    await shutdown()


app = FastAPI(
    title="MarketDrive AI",
    description="Multi-agent financial intelligence system for NSE stocks",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(prediction.router, prefix="/predict",  tags=["Prediction"])
app.include_router(market.router,     prefix="/market",   tags=["Market"])
app.include_router(news.router,       prefix="/news",     tags=["News"])
app.include_router(evaluation.router, prefix="/evaluate", tags=["Evaluation"])
app.include_router(report.router,     prefix="/report",   tags=["Report"])
app.include_router(health.router,                         tags=["Health"])


# ── Admin: Trigger model training on server ────────────────────
async def _run_training_all():
    """Trains models for all 5 stocks using historical Yahoo Finance data."""
    import logging
    from agents.prediction_agent import train_model
    from config.settings import STOCK_CONFIG, TRAINING_DATA_DIR
    import os, yfinance as yf, pandas as pd

    logger = logging.getLogger("admin.train")
    for company, symbol in STOCK_CONFIG.items():
        try:
            logger.info(f"[Admin Train] Downloading data for {company} ({symbol})...")
            df = yf.Ticker(symbol).history(period="2y")
            if df.empty:
                logger.error(f"[Admin Train] No data for {symbol}, skipping.")
                continue
            df.reset_index(inplace=True)
            csv_path = os.path.join(TRAINING_DATA_DIR, f"{company}.csv")
            df.to_csv(csv_path, index=False)
            logger.info(f"[Admin Train] Training model for {company}...")
            result = train_model(company, csv_path)
            logger.info(f"[Admin Train] ✅ {company} trained — {result}")
        except Exception as e:
            logger.error(f"[Admin Train] ❌ {company} failed: {e}")


@app.post("/admin/train", tags=["Admin"])
async def trigger_training(background_tasks: BackgroundTasks):
    """
    Triggers model training for all stocks in the background.
    Call this ONCE after a fresh deployment to Railway.
    Training takes ~5-10 minutes. Check Railway logs for progress.
    """
    background_tasks.add_task(_run_training_all)
    return {
        "status": "training_started",
        "message": "Training all 5 models in background. Check Railway logs for progress. Takes ~5-10 mins.",
        "stocks": ["infosys", "vodafone", "tata", "adani", "yesbank"]
    }