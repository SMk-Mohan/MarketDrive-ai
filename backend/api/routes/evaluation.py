"""
api/routes/evaluation.py
POST /evaluate              → trigger 3:30 PM evaluation manually
GET  /evaluate/history      → past prediction accuracy from global_report
GET  /evaluate/accuracy     → per-stock accuracy summary
"""

import logging
import os
import pandas as pd
from fastapi import APIRouter, HTTPException, BackgroundTasks

from agents.coordinator import run_evaluation
from config.settings import GLOBAL_REPORT_PATH

logger = logging.getLogger(__name__)
router = APIRouter()


# ── POST /evaluate ────────────────────────────────────────────
@router.post("")
async def trigger_evaluation(background_tasks: BackgroundTasks):
    """
    Manually triggers the 3:30 PM evaluation pipeline.
    Fetches actual closing prices via yfinance and logs results.
    Runs in background — returns immediately.
    """
    async def _run():
        try:
            await run_evaluation()
            logger.info("[/evaluate] Evaluation complete")
        except Exception as e:
            logger.error(f"[/evaluate] Error: {e}")

    background_tasks.add_task(_run)
    return {
        "status":  "accepted",
        "message": "Evaluation pipeline started",
    }


# ── GET /evaluate/history ─────────────────────────────────────
@router.get("/history")
async def get_history(stock: str | None = None, limit: int = 30):
    """
    Returns past predictions from global_report.csv.
    Optional ?stock=infosys filter.
    Optional ?limit=N rows (default 30).
    """
    if not os.path.exists(GLOBAL_REPORT_PATH):
        return {"status": "ok", "data": [], "count": 0}

    try:
        df = pd.read_csv(GLOBAL_REPORT_PATH)
        if stock:
            df = df[df["stock"] == stock.lower().strip()]
        df = df.sort_values("date", ascending=False).head(limit)
        return {
            "status": "ok",
            "count":  len(df),
            "data":   df.to_dict(orient="records"),
        }
    except Exception as e:
        logger.error(f"[/evaluate/history] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── GET /evaluate/accuracy ────────────────────────────────────
@router.get("/accuracy")
async def get_accuracy():
    """
    Returns per-stock prediction accuracy summary from global_report.
    Only counts evaluated rows.
    """
    if not os.path.exists(GLOBAL_REPORT_PATH):
        return {"status": "ok", "data": {}}

    try:
        df = pd.read_csv(GLOBAL_REPORT_PATH)
        df = df[df["evaluated"] == True]

        if df.empty:
            return {"status": "ok", "data": {}}

        summary = {}
        for stock, grp in df.groupby("stock"):
            total   = len(grp)
            correct = grp["correct"].sum()
            summary[stock] = {
                "total":       total,
                "correct":     int(correct),
                "wrong":       total - int(correct),
                "accuracy_pct": round(correct / total * 100, 2) if total > 0 else 0.0,
            }

        overall_total   = df.shape[0]
        overall_correct = df["correct"].sum()
        summary["overall"] = {
            "total":        overall_total,
            "correct":      int(overall_correct),
            "wrong":        overall_total - int(overall_correct),
            "accuracy_pct": round(overall_correct / overall_total * 100, 2)
                            if overall_total > 0 else 0.0,
        }

        return {"status": "ok", "data": summary}
    except Exception as e:
        logger.error(f"[/evaluate/accuracy] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))