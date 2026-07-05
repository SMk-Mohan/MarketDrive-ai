"""
api/routes/market.py
GET /market/{ticker}   → live market data for one stock
GET /market/macro      → Nifty, BankNifty, Sensex snapshot
"""

import logging
from fastapi import APIRouter, HTTPException

from agents.market_agent import market_agent_v3, get_all_macro_trends
from core.cache import get_market, set_market
from config.settings import STOCK_CONFIG

logger = logging.getLogger(__name__)
router = APIRouter()

VALID_TICKERS = list(STOCK_CONFIG.keys())


# ── GET /market/macro ─────────────────────────────────────────
@router.get("/macro")
async def get_macro():
    """Returns live Nifty, BankNifty, Sensex trend data."""
    try:
        import asyncio
        from functools import partial
        loop   = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, get_all_macro_trends)
        return {"status": "ok", "data": result}
    except Exception as e:
        logger.error(f"[/market/macro] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── GET /market/proxy/chart ───────────────────────────────────
@router.get("/proxy/chart")
async def proxy_chart(symbol: str, interval: str = "5m", range_val: str = "1d"):
    """
    Fetches OHLC chart data via yfinance (server-side, avoids CORS).
    Returns Yahoo Finance chart-compatible JSON structure.
    """
    import asyncio
    from functools import partial

    def _fetch():
        import yfinance as yf

        period_map = {"1d": "1d", "2d": "5d", "5d": "5d", "1mo": "1mo"}
        period = period_map.get(range_val, "1d")

        # ── Build aggressive fallback chain ──
        attempts = [
            (symbol, period, interval),                   # 1. Primary
            (symbol.replace(".NS", ".BO") if ".NS" in symbol else symbol + ".BO", period, interval), # 2. BSE
            (symbol, "5d", "1h"),                        # 3. Wider intraday
            (symbol, "1mo", "1d"),                       # 4. Daily (most robust)
            ("TMCV.NS", "1mo", "1d")               # 5. Last resort hardcoded
        ]

        df = None
        for sym, per, ivl in attempts:
            try:
                df = yf.Ticker(sym).history(period=per, interval=ivl)
                if not df.empty:
                    # If we fell back to daily data, we need a 'Date' column
                    df = df.reset_index()
                    break
                df = None
            except Exception:
                df = None
                continue

        if df is None or df.empty:
            return None

        df = df.reset_index()

        # ── Safely resolve timestamp column (Datetime for intraday, Date for daily) ──
        if "Datetime" in df.columns:
            ts_col = df["Datetime"]
        elif "Date" in df.columns:
            ts_col = df["Date"]
        else:
            ts_col = df.iloc[:, 0]   # last resort: first column

        timestamps = [int(t.timestamp()) for t in ts_col if hasattr(t, "timestamp")]

        return {
            "chart": {
                "result": [{
                    "timestamp": timestamps,
                    "indicators": {
                        "quote": [{
                            "open":   [float(v) if v == v else None for v in df["Open"]],
                            "high":   [float(v) if v == v else None for v in df["High"]],
                            "low":    [float(v) if v == v else None for v in df["Low"]],
                            "close":  [float(v) if v == v else None for v in df["Close"]],
                            "volume": [int(v) if v == v else 0 for v in df["Volume"]],
                        }]
                    }
                }],
                "error": None
            }
        }

    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, _fetch)
        if result is None:
            raise HTTPException(status_code=404, detail=f"No data for {symbol}")
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[/market/proxy/chart] Error for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── GET /market/{ticker} ──────────────────────────────────────
@router.get("/{ticker}")
async def get_market_data(ticker: str):
    """
    Returns live market data for one stock.
    Checks cache first (5 min TTL). Fetches live if stale.
    """
    company = ticker.lower().strip()
    if company not in VALID_TICKERS:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown ticker '{ticker}'. Valid: {VALID_TICKERS}"
        )

    cached = get_market(company)
    if cached:
        return {"status": "ok", "source": "cache", "data": cached}

    try:
        import asyncio
        from agents.market_agent import run_market_agent
        result = await run_market_agent(company)
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        set_market(company, result)
        return {"status": "ok", "source": "live", "data": result}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[/market/{ticker}] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))