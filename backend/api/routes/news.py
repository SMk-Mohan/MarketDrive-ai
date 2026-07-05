"""
api/routes/news.py
GET /news/{ticker}  → latest news + sentiment for one stock
"""

import logging
from fastapi import APIRouter, HTTPException

from agents.news_agent import run_news_agent
from config.settings import STOCK_CONFIG

logger = logging.getLogger(__name__)
router = APIRouter()

VALID_TICKERS = list(STOCK_CONFIG.keys())


# ── GET /news/{ticker} ────────────────────────────────────────
@router.get("/{ticker}")
async def get_news(ticker: str):
    """
    Fetches and analyzes latest news for one stock via MarketAux + Groq.
    Returns aggregated sentiment, dominant event, and article list.
    Always fetches live — news is not cached (TTL too short to matter).
    """
    company = ticker.lower().strip()
    if company not in VALID_TICKERS:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown ticker '{ticker}'. Valid: {VALID_TICKERS}"
        )

    try:
        result = await run_news_agent(company, historical_summaries=[])
        return {
            "status":  "ok",
            "ticker":  ticker.upper(),
            "data":    result,
        }
    except Exception as e:
        logger.error(f"[/news/{ticker}] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))