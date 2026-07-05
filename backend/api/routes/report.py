"""
api/routes/report.py
GET /report/daily   → today's market_time_report rows
GET /report/global  → full global_report history (paginated)
GET /report/api     → API usage status
"""

import logging
import os
import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from agents.rag_agent import rag_agent
from config.settings import MARKET_TIME_REPORT_PATH, GLOBAL_REPORT_PATH

logger = logging.getLogger(__name__)
router = APIRouter()


# ── GET /report/daily ─────────────────────────────────────────
@router.get("/daily")
async def get_daily_report(stock: str | None = None):
    """
    Returns today's predictions from market_time_report.csv.
    Optional ?stock=infosys to filter one stock.
    """
    if not os.path.exists(MARKET_TIME_REPORT_PATH):
        return {"status": "ok", "data": [], "count": 0}

    try:
        df = pd.read_csv(MARKET_TIME_REPORT_PATH)
        if stock:
            df = df[df["stock"] == stock.lower().strip()]
        df = df.sort_values("time", ascending=False)
        return {
            "status": "ok",
            "count":  len(df),
            "data":   df.to_dict(orient="records"),
        }
    except Exception as e:
        logger.error(f"[/report/daily] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── GET /report/global ────────────────────────────────────────
@router.get("/global")
async def get_global_report(
    stock: str | None = None,
    page:  int = Query(default=1, ge=1),
    limit: int = Query(default=50, ge=1, le=200),
):
    """
    Returns paginated global prediction history.
    Optional ?stock=infosys filter.
    ?page and ?limit for pagination.
    """
    if not os.path.exists(GLOBAL_REPORT_PATH):
        return {"status": "ok", "data": [], "count": 0, "total": 0}

    try:
        df = pd.read_csv(GLOBAL_REPORT_PATH)
        if stock:
            df = df[df["stock"] == stock.lower().strip()]

        df    = df.sort_values("date", ascending=False)
        total = len(df)
        start = (page - 1) * limit
        end   = start + limit
        page_df = df.iloc[start:end]

        return {
            "status":  "ok",
            "total":   total,
            "page":    page,
            "limit":   limit,
            "pages":   (total + limit - 1) // limit,
            "count":   len(page_df),
            "data":    page_df.to_dict(orient="records"),
        }
    except Exception as e:
        logger.error(f"[/report/global] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── GET /report/api ───────────────────────────────────────────
@router.get("/api")
async def get_api_usage():
    """
    Returns today's API usage for MarketAux, Groq, GDELT.
    Useful for monitoring budget on dashboard.
    """
    try:
        result = rag_agent("api_status", {})
        return {"status": "ok", "data": result.get("data", {})}
    except Exception as e:
        logger.error(f"[/report/api] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))