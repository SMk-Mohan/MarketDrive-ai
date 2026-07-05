"""
agents/market_agent.py
Fetches live OHLCV, calculates technical indicators,
macro trends, relative strength, and market breadth.
"""

import asyncio
import logging
import yfinance as yf
import pandas_ta as ta
import pandas as pd
from datetime import datetime
from functools import partial

from config.settings import (
    STOCK_CONFIG,
    SECTOR_INDEX_MAP,
    INDEX_SYMBOLS,
)

logger = logging.getLogger(__name__)


# ── Symbol resolver ───────────────────────────────────────────

def get_symbol(company_name: str) -> str:
    return STOCK_CONFIG.get(
        company_name.lower().strip(),
        company_name.upper() + ".NS"
    )


# ── OHLCV fetcher ─────────────────────────────────────────────

def fetch_ohlcv(symbol: str, period: str = "12mo") -> pd.DataFrame:
    """
    12mo gives EMA200 enough history.
    Flattens MultiIndex columns (yfinance quirk with some versions).
    """
    df = yf.Ticker(symbol).history(period=period)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.dropna(inplace=True)
    if df.empty:
        raise ValueError(f"No OHLCV data returned for {symbol}")
    return df


# ── Safe float ────────────────────────────────────────────────

def safe_float(val, default: float = 0.0) -> float:
    try:
        if val is None or pd.isna(val):
            return default
        return float(val)
    except Exception:
        return default


# ── Current price ─────────────────────────────────────────────

def get_current_price(symbol: str) -> float:
    try:
        info = yf.Ticker(symbol).info
        return float(
            info.get("currentPrice")
            or info.get("regularMarketPrice")
            or 0.0
        )
    except Exception:
        return 0.0


# ── Technical indicators ──────────────────────────────────────

def calculate_indicators(df: pd.DataFrame) -> dict:
    """
    Calculates RSI, MACD, EMA20/50/200, VWAP, ATR, Bollinger Bands.
    Returns last row as flat dict — all keys match FEATURE_COLUMNS.

    FIX: key is 'macd_signal' (not 'macd_signal_line') to match
         FEATURE_COLUMNS in settings.py.
    """
    df = df.copy()
    df["RSI"]   = ta.rsi(df["Close"], length=14)
    macd        = ta.macd(df["Close"], fast=12, slow=26, signal=9)
    df["MACD"]  = macd["MACD_12_26_9"]
    df["MACDs"] = macd["MACDs_12_26_9"]   # signal line
    df["MACDh"] = macd["MACDh_12_26_9"]   # histogram
    df["E20"]   = ta.ema(df["Close"], length=20)
    df["E50"]   = ta.ema(df["Close"], length=50)
    df["E200"]  = ta.ema(df["Close"], length=200)
    df["VWAP"]  = ta.vwap(df["High"], df["Low"],
                           df["Close"], df["Volume"])
    df["ATR"]   = ta.atr(df["High"], df["Low"],
                          df["Close"], length=14)

    l = df.iloc[-1]
    close = safe_float(l.get("Close"),  0.0)
    e20   = safe_float(l.get("E20"),    0.0)

    return {
        # FIX: key renamed to macd_signal (was macd_signal_line)
        "rsi":            safe_float(l.get("RSI"),   50.0),
        "macd":           safe_float(l.get("MACD"),   0.0),
        "macd_signal":    safe_float(l.get("MACDs"),  0.0),  # ← FIX
        "macd_histogram": safe_float(l.get("MACDh"),  0.0),
        "ema_20":         e20,
        "ema_50":         safe_float(l.get("E50"),    0.0),
        "ema_200":        safe_float(l.get("E200"),   0.0),
        "vwap":           safe_float(l.get("VWAP"),   0.0),
        "atr":            safe_float(l.get("ATR"),    0.0),
        "close":          close,
        "open":           safe_float(l.get("Open"),   0.0),
        "high":           safe_float(l.get("High"),   0.0),
        "low":            safe_float(l.get("Low"),    0.0),
        "volume":         int(safe_float(l.get("Volume"), 0)),
        # FIX: added price_vs_ema20 — required by FEATURE_COLUMNS
        "price_vs_ema20": round(close - e20, 4),
    }


# ── Volume analysis ───────────────────────────────────────────

def detect_volume_spike(df: pd.DataFrame) -> dict:
    avg_vol  = safe_float(df["Volume"].rolling(20).mean().iloc[-1], 1.0)
    curr_vol = safe_float(df["Volume"].iloc[-1], 0.0)
    ratio    = round(curr_vol / avg_vol, 2) if avg_vol > 0 else 1.0
    return {
        "current_volume": int(curr_vol),
        "avg_volume_20d": int(avg_vol),
        "volume_ratio":   ratio,
        "volume_spike":   ratio > 1.5,
    }


# ── Trend detection ───────────────────────────────────────────

def detect_trend(indicators: dict) -> tuple[str, int]:
    """
    Score-based trend from 5 signals.
    FIX: also returns trend_score (int 0–5) — required by FEATURE_COLUMNS.
    """
    close  = indicators["close"]
    e20    = indicators["ema_20"]
    e50    = indicators["ema_50"]
    e200   = indicators["ema_200"]
    rsi    = indicators["rsi"]
    macd_h = indicators["macd_histogram"]

    raw_score = 0
    if close > e20:  raw_score += 1
    else:            raw_score -= 1
    if e20   > e50:  raw_score += 1
    else:            raw_score -= 1
    if e50   > e200: raw_score += 1
    else:            raw_score -= 1
    if rsi > 55:     raw_score += 1
    elif rsi < 45:   raw_score -= 1
    if macd_h > 0:   raw_score += 1
    else:            raw_score -= 1

    # trend_score as 0–5 (ML-friendly, non-negative)
    trend_score = raw_score + 5  # range: 0–10, clamp to 0–5 display
    trend_score = max(0, min(10, trend_score))

    label_map = {
        10: "Strong Bullish", 9: "Strong Bullish",
        8:  "Bullish",        7: "Bullish",
        6:  "Weak Bullish",   5: "Sideways",
        4:  "Weak Bearish",   3: "Weak Bearish",
        2:  "Bearish",        1: "Bearish",
        0:  "Strong Bearish",
    }
    return label_map.get(trend_score, "Sideways"), trend_score


def detect_macd_signal(indicators: dict) -> str:
    if indicators["macd"] > indicators["macd_signal"]:   # FIX: key
        return "Bullish"
    if indicators["macd"] < indicators["macd_signal"]:
        return "Bearish"
    return "Neutral"


def detect_momentum(indicators: dict) -> str:
    rsi = indicators["rsi"]
    if rsi >= 65: return "Strong"
    if rsi >= 55: return "Moderate"
    if rsi >= 45: return "Weak"
    return "Oversold"


def detect_volatility_regime(indicators: dict) -> str:
    atr   = indicators["atr"]
    close = indicators["close"]
    pct   = (atr / close) * 100 if close > 0 else 0
    if pct < 1.5: return "Low"
    if pct < 3.0: return "Medium"
    return "High"


# ── Support & Resistance ──────────────────────────────────────

def calculate_support_resistance(df: pd.DataFrame) -> dict:
    try:
        recent          = df.tail(20)
        prev            = df.tail(21).head(20)
        support         = round(float(recent["Low"].min()),  2)
        resistance      = round(float(recent["High"].max()), 2)
        close           = round(float(df["Close"].iloc[-1]), 2)
        prev_resistance = round(float(prev["High"].max()),   2)
        prev_support    = round(float(prev["Low"].min()),    2)

        sr_signal = "None"
        if close > prev_resistance: sr_signal = "Breakout"
        if close < prev_support:    sr_signal = "Breakdown"

        return {
            "support":    support,
            "resistance": resistance,
            "sr_signal":  sr_signal,
        }
    except Exception as e:
        logger.error(f"[Market Agent] S/R error: {e}")
        return {"support": 0.0, "resistance": 0.0, "sr_signal": "None"}


# ── Macro index trends ────────────────────────────────────────

def get_index_trend(symbol: str, name: str) -> dict:
    try:
        df = yf.Ticker(symbol).history(period="3mo")
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.dropna(inplace=True)
        df["EMA_20"] = ta.ema(df["Close"], length=20)
        df["RSI"]    = ta.rsi(df["Close"], length=14)
        l     = df.iloc[-1]
        close = safe_float(l.get("Close"), 0.0)
        ema20 = safe_float(l.get("EMA_20"), 0.0)
        rsi   = safe_float(l.get("RSI"),   50.0)

        if close > ema20 and rsi > 55:   trend = "Bullish"
        elif close < ema20 and rsi < 45: trend = "Bearish"
        else:                             trend = "Neutral"

        return {
            f"{name}_price": round(close, 2),
            f"{name}_trend": trend,
            f"{name}_rsi":   round(rsi,   2),
        }
    except Exception as e:
        logger.error(f"[Market Agent] {name} index error: {e}")
        return {
            f"{name}_price": 0.0,
            f"{name}_trend": "Unknown",
            f"{name}_rsi":   0.0,
        }


def get_all_macro_trends() -> dict:
    result = {}
    for name, symbol in INDEX_SYMBOLS.items():
        result.update(get_index_trend(symbol, name))
    return result


# ── Relative strength ─────────────────────────────────────────

def get_relative_strength(symbol: str) -> dict:
    try:
        sector_sym = SECTOR_INDEX_MAP.get(symbol, "^NSEI")

        def pct_return(df, days):
            df = df.dropna()
            if len(df) < days: return 0.0
            return (df["Close"].iloc[-1] - df["Close"].iloc[-days]) \
                   / df["Close"].iloc[-days] * 100

        stock_df  = yf.Ticker(symbol).history(period="3mo")
        nifty_df  = yf.Ticker("^NSEI").history(period="3mo")
        sector_df = yf.Ticker(sector_sym).history(period="3mo")

        s5,  s20 = pct_return(stock_df,  5),  pct_return(stock_df,  20)
        n5,  n20 = pct_return(nifty_df,  5),  pct_return(nifty_df,  20)
        x5,  x20 = pct_return(sector_df, 5),  pct_return(sector_df, 20)

        def label(sr, br):
            d = sr - br
            if d >  1.0: return "Outperforming"
            if d < -1.0: return "Underperforming"
            return "In-line"

        return {
            "stock_return_5d":  round(s5,  2),
            "stock_return_20d": round(s20, 2),
            "vs_nifty_5d":      label(s5,  n5),
            "vs_nifty_20d":     label(s20, n20),
            "vs_sector_5d":     label(s5,  x5),
            "vs_sector_20d":    label(s20, x20),
        }
    except Exception as e:
        logger.error(f"[Market Agent] Relative strength error: {e}")
        return {
            "stock_return_5d":  0.0, "stock_return_20d": 0.0,
            "vs_nifty_5d":  "Unknown", "vs_nifty_20d":  "Unknown",
            "vs_sector_5d": "Unknown", "vs_sector_20d": "Unknown",
        }


# ── Market breadth ────────────────────────────────────────────

def get_market_breadth(symbol: str) -> dict:
    try:
        sector_sym = SECTOR_INDEX_MAP.get(symbol, "^NSEI")

        def trend_5d(df):
            df = df.dropna()
            if len(df) < 2: return "Flat"
            ret = (df["Close"].iloc[-1] - df["Close"].iloc[0]) \
                  / df["Close"].iloc[0] * 100
            if ret >  0.5: return "Rising"
            if ret < -0.5: return "Falling"
            return "Flat"

        sector_df    = yf.Ticker(sector_sym).history(period="5d")
        nifty_df     = yf.Ticker("^NSEI").history(period="5d")
        banknifty_df = yf.Ticker("^NSEBANK").history(period="5d")
        sensex_df    = yf.Ticker("^BSESN").history(period="5d")

        trends = [
            trend_5d(sector_df),
            trend_5d(nifty_df),
            trend_5d(banknifty_df),
            trend_5d(sensex_df),
        ]
        rising  = sum(1 for t in trends if t == "Rising")
        falling = sum(1 for t in trends if t == "Falling")

        if rising  >= 3: breadth = "Strong"
        elif falling >= 3: breadth = "Weak"
        else:              breadth = "Mixed"

        return {
            "sector_trend_5d":    trends[0],
            "nifty_trend_5d":     trends[1],
            "banknifty_trend_5d": trends[2],
            "sensex_trend_5d":    trends[3],
            "market_breadth":     breadth,
        }
    except Exception as e:
        logger.error(f"[Market Agent] Breadth error: {e}")
        return {
            "sector_trend_5d":    "Unknown",
            "nifty_trend_5d":     "Unknown",
            "banknifty_trend_5d": "Unknown",
            "sensex_trend_5d":    "Unknown",
            "market_breadth":     "Unknown",
        }


# ── Main Market Agent (sync) ──────────────────────────────────

def market_agent_v3(company_name: str) -> dict:
    """
    Fetches and assembles all market intelligence for one stock.

    FIX: outputs macd_signal (not macd_signal_line)
    FIX: outputs trend_score and price_vs_ema20
    Both required by FEATURE_COLUMNS / prediction_agent.

    Args:
        company_name: e.g. "infosys"

    Returns:
        Full market intelligence dict for coordinator + prediction agent
    """
    try:
        symbol = get_symbol(company_name)
        logger.info(f"[Market Agent V3] Fetching → {symbol}")

        df            = fetch_ohlcv(symbol)
        current_price = get_current_price(symbol)
        indicators    = calculate_indicators(df)
        volume_data   = detect_volume_spike(df)
        sr_data       = calculate_support_resistance(df)
        macro_trends  = get_all_macro_trends()
        rs_data       = get_relative_strength(symbol)
        breadth_data  = get_market_breadth(symbol)

        trend, trend_score = detect_trend(indicators)   # FIX: unpack tuple
        macd_sig           = detect_macd_signal(indicators)
        momentum           = detect_momentum(indicators)
        vol_regime         = detect_volatility_regime(indicators)

        atr_pct = round(
            (indicators["atr"] / indicators["close"]) * 100
            if indicators["close"] > 0 else 0.0, 4
        )

        result = {
            # Identity
            "symbol":   symbol,
            "company":  company_name,
            "timestamp": datetime.now().isoformat(),

            # Price (OHLCV)
            "current_price": current_price or indicators["close"],
            "open":          indicators["open"],
            "high":          indicators["high"],
            "low":           indicators["low"],
            "close":         indicators["close"],

            # Trend labels
            "trend":             trend,
            "macd_signal_label": macd_sig,
            "momentum":          momentum,
            "volatility_regime": vol_regime,

            # Indicators — keys match FEATURE_COLUMNS exactly
            "rsi":            indicators["rsi"],
            "macd":           indicators["macd"],
            "macd_signal":    indicators["macd_signal"],    # ← FIX
            "macd_histogram": indicators["macd_histogram"],
            "ema_20":         indicators["ema_20"],
            "ema_50":         indicators["ema_50"],
            "ema_200":        indicators["ema_200"],
            "vwap":           indicators["vwap"],
            "atr":            indicators["atr"],
            "atr_pct":        atr_pct,

            # Derived features — required by FEATURE_COLUMNS
            "trend_score":    trend_score,                  # ← FIX
            "price_vs_ema20": indicators["price_vs_ema20"], # ← FIX

            # Volume
            "volume_spike":   volume_data["volume_spike"],
            "volume_ratio":   volume_data["volume_ratio"],
            "current_volume": volume_data["current_volume"],
            "avg_volume_20d": volume_data["avg_volume_20d"],

            # Support / Resistance
            **sr_data,

            # Macro
            **macro_trends,

            # Relative strength
            **rs_data,

            # Market breadth
            **breadth_data,
        }

        logger.info(
            f"[Market Agent V3] {symbol} done — "
            f"trend={trend} | rsi={indicators['rsi']:.1f} | "
            f"trend_score={trend_score} | breadth={breadth_data['market_breadth']}"
        )
        return result

    except Exception as e:
        logger.error(f"[Market Agent V3] Error for {company_name}: {e}")
        return {"error": str(e), "company": company_name}


# ── Async wrapper for coordinator ─────────────────────────────

async def run_market_agent(company_name: str) -> dict:
    """
    Async wrapper — runs sync market_agent_v3 in thread pool
    so it doesn't block FastAPI's event loop.
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        partial(market_agent_v3, company_name)
    )
