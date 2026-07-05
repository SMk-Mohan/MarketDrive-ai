"""
agents/coordinator.py
LangGraph orchestration — coordinates all 6 agents.
Handles morning prediction, trigger-based re-prediction,
evening evaluation, and end-of-day retraining.
"""

import os
import asyncio
import logging
import time
import pandas as pd
from datetime import datetime, date
from typing import TypedDict, Optional, Dict, Any

from langgraph.graph import StateGraph, END

from config.settings import (
    STOCK_CONFIG,
    STOCK_DISPLAY_TICKER,
    TRAINING_DATA_DIR,
    MARKET_TIME_REPORT_PATH,
    GLOBAL_REPORT_PATH,
)
from agents.news_agent       import news_agent_v3
from agents.market_agent     import market_agent_v3
from agents.prediction_agent import predict, train_model
from agents.risk_agent       import risk_agent
from agents.rag_agent        import rag_agent

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
# STATE SCHEMA
# ══════════════════════════════════════════════════════════════

class GraphState(TypedDict):
    # Input
    company_name:          str
    symbol:                str

    # Agent outputs
    news_result:           Optional[Dict]
    market_result:         Optional[Dict]
    prediction_result:     Optional[Dict]
    risk_result:           Optional[Dict]
    rag_explanation:       Optional[str]

    # Coordinator decisions
    trigger_detected:      bool
    trigger_reason:        str
    cycle_count:           int
    market_closed:         bool
    skip_news:             bool
    morning_market_result: Optional[Dict]

    # Final output
    final_output:          Optional[Dict]
    adjusted_confidence:   Optional[float]
    signals_conflicting:   bool


# ══════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════

def detect_conflicts(
    prediction:    str,
    news_result:   dict,
    market_result: dict,
) -> tuple[bool, list]:
    """
    Checks if news sentiment contradicts technical prediction.
    Returns (conflicting: bool, conflict_descriptions: list)
    """
    sentiment = news_result.get("sentiment_score", 0.0)
    nifty     = market_result.get("nifty_trend",    "Neutral")
    breadth   = market_result.get("market_breadth", "Mixed")
    conflicts = []

    if prediction == "Bullish" and sentiment < -0.4:
        conflicts.append("Negative news vs Bullish prediction")
    if prediction == "Bearish" and sentiment > 0.4:
        conflicts.append("Positive news vs Bearish prediction")
    if prediction == "Bullish" and nifty == "Bearish":
        conflicts.append("Nifty Bearish vs Bullish prediction")
    if prediction == "Bearish" and nifty == "Bullish":
        conflicts.append("Nifty Bullish vs Bearish prediction")
    if prediction == "Bullish" and breadth == "Weak":
        conflicts.append("Weak market breadth vs Bullish prediction")

    return len(conflicts) > 0, conflicts


def adjust_confidence(
    base_confidence:   float,
    conflicting:       bool,
    conflicts:         list,
    volatility_regime: str,
    volume_spike:      bool,
    prediction:        str,
    market_result:     dict,
    news_result:       dict,
) -> float:
    """
    Adjusts raw model confidence based on cross-agent signal alignment.
    Clamps result to [10.0, 95.0].
    """
    adj = base_confidence

    if conflicting:
        adj -= 15
        logger.info(f"  [Coordinator] Confidence −15: conflicts {conflicts}")

    if volatility_regime == "High":
        adj -= 10
        logger.info("  [Coordinator] Confidence −10: high volatility")

    if prediction == "Bullish" and market_result.get("market_breadth") == "Weak":
        adj -= 8
        logger.info("  [Coordinator] Confidence −8: weak breadth")

    nifty     = market_result.get("nifty_trend",     "Neutral")
    banknifty = market_result.get("banknifty_trend", "Neutral")
    sensex    = market_result.get("sensex_trend",    "Neutral")

    if nifty == banknifty == sensex == prediction:
        adj += 5
        logger.info("  [Coordinator] Confidence +5: all indices aligned")

    if volume_spike:
        adj += 3
        logger.info("  [Coordinator] Confidence +3: volume confirmation")

    sentiment = news_result.get("sentiment_score", 0.0)
    if prediction == "Bullish" and sentiment > 0.5:
        adj += 4
    elif prediction == "Bearish" and sentiment < -0.5:
        adj += 4

    return round(max(10.0, min(95.0, adj)), 1)


def check_triggers(
    market_result:  dict,
    news_result:    dict,
    morning_market: dict,
) -> tuple[bool, str]:
    """
    Checks all 4 mid-day trigger conditions.
    Returns (triggered: bool, reason: str)
    """
    # 1. Volume spike
    if market_result.get("volume_ratio", 1.0) > 2.5:
        return True, "VolSpike"

    # 2. Breaking high-impact news
    if news_result.get("impact_score", 0.0) > 0.7:
        return True, "HighImpactNews"

    # 3. Nifty sudden move > 1%
    curr_nifty    = market_result.get("nifty_price",  0.0)
    morning_nifty = morning_market.get("nifty_price", 0.0)
    if morning_nifty > 0:
        nifty_move = abs((curr_nifty - morning_nifty) / morning_nifty * 100)
        if nifty_move > 1.0:
            return True, "NiftyMove"

    # 4. Breakout / Breakdown
    sr_signal = market_result.get("sr_signal", "None")
    if sr_signal in ["Breakout", "Breakdown"]:
        return True, f"SR_{sr_signal}"

    return False, "None"


# ══════════════════════════════════════════════════════════════
# LANGGRAPH NODES
# ══════════════════════════════════════════════════════════════

def news_agent_node(state: GraphState) -> GraphState:
    """
    Node 1: Calls News Agent.
    FIX: Uses fixed news_agent_v3 which already returns aggregated dict.
    Skips if API budget is exhausted.
    """
    logger.info(f"\n[Node: News Agent] {state['company_name'].upper()}")

    # Check API budget
    remaining = rag_agent("api_remaining", {"service": "marketaux"})
    if remaining.get("remaining", 99) < 3:
        logger.warning("  [News Agent] MarketAux budget low — using zero sentiment")
        state["skip_news"] = True
        state["news_result"] = {
            "articles":        [],
            "news_count":      0,
            "sentiment_score": 0.0,
            "impact_score":    0.0,
            "relevance_score": 0.0,
            "dominant_event":  "Other",
        }
        return state

    try:
        # FIX: news_agent_v3 now returns aggregated dict directly
        result = news_agent_v3(state["company_name"], historical_summaries=[])
        state["news_result"] = result
        rag_agent("log_api_call", {"service": "marketaux", "calls": 1})
    except Exception as e:
        logger.error(f"  [News Agent] Error: {e}")
        state["news_result"] = {
            "articles":        [],
            "news_count":      0,
            "sentiment_score": 0.0,
            "impact_score":    0.0,
            "relevance_score": 0.0,
            "dominant_event":  "Other",
        }

    logger.info(
        f"  Sentiment: {state['news_result']['sentiment_score']:.3f} | "
        f"Articles: {state['news_result']['news_count']} | "
        f"Event: {state['news_result']['dominant_event']}"
    )
    return state


def market_agent_node(state: GraphState) -> GraphState:
    """Node 2: Calls Market Agent V3."""
    logger.info(f"\n[Node: Market Agent] {state['company_name'].upper()}")
    try:
        result = market_agent_v3(state["company_name"])
        state["market_result"] = result
    except Exception as e:
        logger.error(f"  [Market Agent] Error: {e}")
        state["market_result"] = {"error": str(e)}
    return state


def prediction_agent_node(state: GraphState) -> GraphState:
    """
    Node 3: Assembles feature dict from News + Market outputs
    and calls Prediction Agent.

    FIX: macd field maps to market.get("macd") not macd_histogram.
    FIX: all FEATURE_COLUMNS present including trend_score, price_vs_ema20, news_count.
    """
    logger.info(f"\n[Node: Prediction Agent] {state['company_name'].upper()}")

    news   = state.get("news_result",   {}) or {}
    market = state.get("market_result", {}) or {}

    live_features = {
        # Price
        "open":            market.get("open",            0.0),
        "high":            market.get("high",            0.0),
        "low":             market.get("low",             0.0),
        "close":           market.get("close",           market.get("current_price", 0.0)),

        # Indicators — FIX: macd maps to "macd" not "macd_histogram"
        "rsi":             market.get("rsi",             50.0),
        "macd":            market.get("macd",            0.0),   # ← FIX
        "macd_signal":     market.get("macd_signal",     0.0),
        "macd_histogram":  market.get("macd_histogram",  0.0),

        # Trend
        "ema_20":          market.get("ema_20",          0.0),
        "ema_50":          market.get("ema_50",          0.0),
        "ema_200":         market.get("ema_200",         0.0),

        # Volatility
        "atr":             market.get("atr",             0.0),
        "atr_pct":         market.get("atr_pct",         0.0),

        # Volume
        "volume_ratio":    market.get("volume_ratio",    1.0),

        # Derived — FIX: these were missing before market_agent fix
        "trend_score":     market.get("trend_score",     0),
        "price_vs_ema20":  market.get("price_vs_ema20",  0.0),

        # Macro categoricals (raw — encoder handles them)
        "nifty_trend":     market.get("nifty_trend",     "Neutral"),
        "banknifty_trend": market.get("banknifty_trend", "Neutral"),
        "sensex_trend":    market.get("sensex_trend",    "Neutral"),

        # News features — FIX: news_count was missing before news_agent fix
        "sentiment_score": news.get("sentiment_score",   0.0),
        "impact_score":    news.get("impact_score",      0.0),
        "relevance_score": news.get("relevance_score",   0.0),
        "news_count":      news.get("news_count",        0),

        # Dominant event categorical (raw — encoder handles it)
        "dominant_event":  news.get("dominant_event",    "Other"),

        # Trigger count increments on re-predictions
        "trigger_count":   state.get("cycle_count",      0),
    }

    try:
        pred_result = predict(state["company_name"], live_features)
        state["prediction_result"] = pred_result
    except Exception as e:
        logger.error(f"  [Prediction Agent] Error: {e}")
        close = live_features["close"]
        state["prediction_result"] = {
            "company":         state["company_name"],
            "prediction":      "Neutral",
            "confidence":      33.0,
            "probabilities":   {"Bullish": 33.0, "Neutral": 34.0, "Bearish": 33.0},
            "current_price":   close,
            "price_range_low":  round(close * 0.99, 2),
            "price_range_high": round(close * 1.01, 2),
            "top_features":    [],
            "reason":          f"Prediction error: {e}",
            "timestamp":       datetime.now().isoformat(),
        }

    logger.info(
        f"  Prediction: {state['prediction_result']['prediction']} | "
        f"Confidence: {state['prediction_result']['confidence']:.1f}%"
    )
    return state


def coordinator_node(state: GraphState) -> GraphState:
    """
    Node 4: Cross-agent conflict detection + confidence adjustment.
    FIX: does NOT mutate cycle_count here (routing fn must be pure).
    """
    logger.info(f"\n[Node: Coordinator] {state['company_name'].upper()}")

    news   = state.get("news_result",       {}) or {}
    market = state.get("market_result",     {}) or {}
    pred   = state.get("prediction_result", {}) or {}

    prediction      = pred.get("prediction", "Neutral")
    base_confidence = pred.get("confidence", 33.0)

    conflicting, conflicts = detect_conflicts(prediction, news, market)
    state["signals_conflicting"] = conflicting

    if conflicting:
        logger.warning(f"  ⚠️  Conflicts: {conflicts}")

    adjusted_conf = adjust_confidence(
        base_confidence   = base_confidence,
        conflicting       = conflicting,
        conflicts         = conflicts,
        volatility_regime = market.get("volatility_regime", "Medium"),
        volume_spike      = market.get("volume_spike",      False),
        prediction        = prediction,
        market_result     = market,
        news_result       = news,
    )
    state["adjusted_confidence"] = adjusted_conf

    # Trigger check for monitoring cycles (cycle_count > 0)
    if state.get("cycle_count", 0) > 0:
        morning_market = state.get("morning_market_result") or market
        triggered, reason = check_triggers(market, news, morning_market)
        state["trigger_detected"] = triggered
        state["trigger_reason"]   = reason
        if triggered:
            logger.info(f"  🔔 TRIGGER: {reason}")
    else:
        state["trigger_detected"]    = False
        state["trigger_reason"]      = "None"
        state["morning_market_result"] = market  # save baseline

    logger.info(
        f"  Confidence: {base_confidence:.1f}% → {adjusted_conf:.1f}% | "
        f"Conflicting: {conflicting}"
    )
    return state


def risk_agent_node(state: GraphState) -> GraphState:
    """Node 5: Calls Risk Agent."""
    logger.info(f"\n[Node: Risk Agent] {state['company_name'].upper()}")

    news   = state.get("news_result",       {}) or {}
    market = state.get("market_result",     {}) or {}
    pred   = state.get("prediction_result", {}) or {}

    payload = {
        "company":            state["company_name"],
        "prediction":         pred.get("prediction",    "Neutral"),
        "confidence":         state.get("adjusted_confidence",
                              pred.get("confidence",    33.0)),
        "probabilities":      pred.get("probabilities",
                              {"Bullish":33.0,"Neutral":34.0,"Bearish":33.0}),
        "volatility_regime":  market.get("volatility_regime", "Medium"),
        "volume_ratio":       market.get("volume_ratio",       1.0),
        "volume_spike":       market.get("volume_spike",       False),
        "market_breadth":     market.get("market_breadth",     "Mixed"),
        "sr_signal":          market.get("sr_signal",          "None"),
        "sentiment_score":    news.get("sentiment_score",      0.0),
        "impact_score":       news.get("impact_score",         0.0),
        "dominant_event":     news.get("dominant_event",       "Other"),
        "nifty_trend":        market.get("nifty_trend",        "Neutral"),
        "banknifty_trend":    market.get("banknifty_trend",    "Neutral"),
        "sensex_trend":       market.get("sensex_trend",       "Neutral"),
        "signals_conflicting": state.get("signals_conflicting", False),
        "atr_pct":            market.get("atr_pct",            0.0),
    }

    state["risk_result"] = risk_agent(payload)
    return state


def rag_agent_node(state: GraphState) -> GraphState:
    """
    Node 6: RAG Agent — generates explanation + logs prediction.
    FIX: rag_agent is properly imported, not called as undefined.
    """
    logger.info(f"\n[Node: RAG Agent] {state['company_name'].upper()}")

    news   = state.get("news_result",       {}) or {}
    market = state.get("market_result",     {}) or {}
    pred   = state.get("prediction_result", {}) or {}

    prediction = pred.get("prediction", "Neutral")
    confidence = state.get("adjusted_confidence", pred.get("confidence", 33.0))

    features_for_rag = {
        "rsi":             market.get("rsi",             50.0),
        "macd_signal":     market.get("macd_signal",     0.0),
        "volume_ratio":    market.get("volume_ratio",    1.0),
        "trend_score":     market.get("trend_score",     0),
        "nifty_trend":     market.get("nifty_trend",     "Neutral"),
        "banknifty_trend": market.get("banknifty_trend", "Neutral"),
        "sensex_trend":    market.get("sensex_trend",    "Neutral"),
        "sentiment_score": news.get("sentiment_score",   0.0),
        "dominant_event":  news.get("dominant_event",    "Other"),
        "trigger_reason":  state.get("trigger_reason",   "None"),
        "correct":         None,  # unknown until evening evaluation
    }

    # Generate LLM explanation
    try:
        explanation_result = rag_agent("explain", {
            "stock":      state["company_name"],
            "features":   features_for_rag,
            "prediction": prediction,
            "confidence": confidence,
            "shap_top":   pred.get("top_features", []),
        })
        state["rag_explanation"] = explanation_result.get(
            "explanation",
            pred.get("reason", f"{state['company_name']} Predicted {prediction}")
        )
    except Exception as e:
        logger.error(f"  [RAG Agent] Explanation error: {e}")
        state["rag_explanation"] = pred.get(
            "reason", 
            f"{state['company_name']} Predicted {prediction}"
        )

    # Log prediction to market_time_report
    try:
        rag_agent("log_prediction", {
            "stock":            state["company_name"],
            "symbol":           state["symbol"],
            "prediction":       prediction,
            "confidence":       confidence,
            "price_range_low":  pred.get("price_range_low",  0.0),
            "price_range_high": pred.get("price_range_high", 0.0),
            "features":         features_for_rag,
            "trigger_count":    state.get("cycle_count", 0),
            "trigger_reason":   state.get("trigger_reason", "None"),
            "is_morning":       state.get("cycle_count", 0) == 0,
        })
        rag_agent("log_api_call", {"service": "groq", "calls": 1})
    except Exception as e:
        logger.error(f"  [RAG Agent] Log error: {e}")

    logger.info(f"  Explanation: {state['rag_explanation'][:80]}...")
    return state


def assemble_output_node(state: GraphState) -> GraphState:
    """Node 7: Packages all agent outputs into final API response dict."""
    logger.info(f"\n[Node: Assemble Output] {state['company_name'].upper()}")

    news   = state.get("news_result",       {}) or {}
    market = state.get("market_result",     {}) or {}
    pred   = state.get("prediction_result", {}) or {}
    risk   = state.get("risk_result",       {}) or {}

    state["final_output"] = {
        # Identity
        "company":    state["company_name"],
        "ticker":     STOCK_DISPLAY_TICKER.get(state["symbol"], state["symbol"]),
        "symbol":     state["symbol"],
        "timestamp":  datetime.now().isoformat(),

        # Prediction
        "prediction":       pred.get("prediction",    "Neutral"),
        "confidence":       state.get("adjusted_confidence",
                            pred.get("confidence",    33.0)),
        "probabilities":    pred.get("probabilities", {}),

        # Price
        "current_price":    market.get("current_price",  0.0),
        "price_range_low":  pred.get("price_range_low",  0.0),
        "price_range_high": pred.get("price_range_high", 0.0),

        # Risk
        "risk":             risk.get("risk",            "Medium"),
        "risk_score":       risk.get("risk_score",       0.5),
        "risk_factors":     risk.get("risk_factors",     []),
        "trade_suggestion": risk.get("trade_suggestion", ""),

        # Explanation
        "explanation":     state.get("rag_explanation", ""),
        "top_features":    pred.get("top_features",    []),
        "reason":          pred.get("reason",          ""),

        # Trigger
        "trigger_count":   state.get("cycle_count",    0),
        "trigger_reason":  state.get("trigger_reason", "None"),

        # Signals for dashboard
        "key_signals": {
            "rsi":             market.get("rsi",            0),
            "macd":            market.get("macd",           0),
            "macd_signal":     market.get("macd_signal",    0),
            "ema_20":          market.get("ema_20",         0),
            "ema_50":          market.get("ema_50",         0),
            "trend":           market.get("trend",          ""),
            "trend_score":     market.get("trend_score",    0),
            "nifty_trend":     market.get("nifty_trend",    ""),
            "banknifty_trend": market.get("banknifty_trend",""),
            "sensex_trend":    market.get("sensex_trend",   ""),
            "volume_spike":    market.get("volume_spike",   False),
            "volume_ratio":    market.get("volume_ratio",   1.0),
            "market_breadth":  market.get("market_breadth", ""),
            "sentiment_score": news.get("sentiment_score",  0.0),
            "dominant_event":  news.get("dominant_event",   "Other"),
            "news_count":      news.get("news_count",       0),
            "articles":        news.get("articles",         []),
            "sr_signal":       market.get("sr_signal",      "None"),
        },

        "signals_conflicting": state.get("signals_conflicting", False),
        "model_accuracy":      pred.get("model_accuracy", 0.0),
        "model_trained_at":    pred.get("model_trained_at", ""),
    }

    logger.info(
        f"  ✅ {state['final_output']['prediction']} | "
        f"Conf: {state['final_output']['confidence']:.1f}% | "
        f"Risk: {state['final_output']['risk']}"
    )
    return state


# ══════════════════════════════════════════════════════════════
# ROUTING — must be pure, no state mutation
# ══════════════════════════════════════════════════════════════

def should_continue(state: GraphState) -> str:
    """
    LangGraph conditional routing function.
    FIX: does NOT mutate state — cycle_count incremented in pipeline wrapper.
    """
    if state.get("market_closed", False):
        return "end"
    if state.get("trigger_detected", False):
        return "re_predict"
    return "assemble"


# ══════════════════════════════════════════════════════════════
# BUILD GRAPH
# ══════════════════════════════════════════════════════════════

def build_coordinator_graph():
    graph = StateGraph(GraphState)

    graph.add_node("news_agent",       news_agent_node)
    graph.add_node("market_agent",     market_agent_node)
    graph.add_node("prediction_agent", prediction_agent_node)
    graph.add_node("coordinator",      coordinator_node)
    graph.add_node("risk_agent",       risk_agent_node)
    graph.add_node("rag_agent",        rag_agent_node)
    graph.add_node("assemble_output",  assemble_output_node)

    graph.set_entry_point("news_agent")

    graph.add_edge("news_agent",       "market_agent")
    graph.add_edge("market_agent",     "prediction_agent")
    graph.add_edge("prediction_agent", "coordinator")
    graph.add_edge("coordinator",      "risk_agent")
    graph.add_edge("risk_agent",       "rag_agent")

    # FIX: cycle_count incremented in run_pipeline, not inside routing fn
    graph.add_conditional_edges(
        "rag_agent",
        should_continue,
        {
            "re_predict": "news_agent",
            "assemble":   "assemble_output",
            "end":        END,
        }
    )
    graph.add_edge("assemble_output", END)

    return graph.compile()


_graph = build_coordinator_graph()


# ══════════════════════════════════════════════════════════════
# PUBLIC API
# FIX: renamed run_prediction → run_pipeline to avoid conflict
#      with prediction_agent.run_prediction
# ══════════════════════════════════════════════════════════════

async def run_pipeline(company_name: str) -> dict:
    """
    Runs the full prediction pipeline for one stock (async).
    Called by scheduler at 9:15 AM and on event triggers.

    Args:
        company_name: e.g. "infosys"

    Returns:
        final_output dict
    """
    symbol = STOCK_CONFIG.get(company_name.lower(), "")
    if not symbol:
        logger.error(f"[Coordinator] Unknown company: {company_name}")
        return {"error": f"Unknown company: {company_name}"}

    initial_state: GraphState = {
        "company_name":          company_name,
        "symbol":                symbol,
        "news_result":           None,
        "market_result":         None,
        "prediction_result":     None,
        "risk_result":           None,
        "rag_explanation":       None,
        "trigger_detected":      False,
        "trigger_reason":        "None",
        "cycle_count":           0,
        "market_closed":         False,
        "skip_news":             False,
        "morning_market_result": None,
        "final_output":          None,
        "adjusted_confidence":   None,
        "signals_conflicting":   False,
    }

    logger.info(f"\n{'='*60}")
    logger.info(f" COORDINATOR — {company_name.upper()} [{symbol}]")
    logger.info(f"{'='*60}")

    loop         = asyncio.get_event_loop()
    # FIX: LangGraph invoke is sync — run in executor so we don't block event loop
    final_state  = await loop.run_in_executor(
        None, _graph.invoke, initial_state
    )

    # FIX: cycle_count incremented here after graph returns, not inside routing fn
    result = final_state.get("final_output", {})

    # Handle trigger re-prediction cycles
    max_cycles = 3
    cycle      = 0
    while (
        final_state.get("trigger_detected") and
        not final_state.get("market_closed") and
        cycle < max_cycles
    ):
        cycle += 1
        logger.info(f"  [Coordinator] Re-prediction cycle {cycle} — trigger: {final_state.get('trigger_reason')}")
        final_state["cycle_count"] = cycle
        final_state["trigger_detected"] = False
        final_state = await loop.run_in_executor(
            None, _graph.invoke, final_state
        )
        result = final_state.get("final_output", result)

    return result


async def run_all_predictions() -> dict:
    """
    Runs pipeline for all 5 stocks. Called at 9:15 AM by scheduler.
    Returns dict keyed by company name.
    """
    all_results = {}
    for company in STOCK_CONFIG.keys():
        try:
            result = await run_pipeline(company)
            all_results[company] = result
            await asyncio.sleep(2)   # small gap between stocks
        except Exception as e:
            logger.error(f"[Coordinator] {company} failed: {e}")
            all_results[company] = {"error": str(e)}

    logger.info(f"\n{'='*60}\n MORNING PREDICTION SUMMARY")
    for company, result in all_results.items():
        logger.info(
            f" {company.upper():12} → "
            f"{result.get('prediction','?'):8} | "
            f"Conf: {result.get('confidence',0):.1f}% | "
            f"Risk: {result.get('risk','?')}"
        )
    logger.info(f"{'='*60}")
    return all_results


async def run_evaluation(actual_closes: dict | None = None) -> None:
    """
    Evening evaluation at 3:30 PM.
    Compares morning predictions against actual closes.
    Fetches actual prices via yfinance if actual_closes not provided.
    """
    logger.info(f"\n{'='*60}\n EVENING EVALUATION (3:30 PM)\n{'='*60}")

    if actual_closes is None:
        import yfinance as yf
        actual_closes = {}
        for company, symbol in STOCK_CONFIG.items():
            try:
                df = yf.Ticker(symbol).history(period="1d")
                if not df.empty:
                    actual_closes[company] = float(df["Close"].iloc[-1])
            except Exception as e:
                logger.error(f"  {company}: price fetch error: {e}")

    if not os.path.exists(MARKET_TIME_REPORT_PATH):
        logger.warning("  No market_time_report.csv found — skipping evaluation")
        return

    df    = pd.read_csv(MARKET_TIME_REPORT_PATH)
    today = datetime.now().strftime("%Y-%m-%d")

    for company, actual_close in actual_closes.items():
        row = df[(df["stock"] == company) & (df["date"] == today)]
        if row.empty:
            logger.warning(f"  {company}: no prediction found for {today}")
            continue

        morning_pred  = row["final_prediction"].values[0]
        morning_price = float(row.get("current_price", row.get("price_range_low", [0])).values[0])

        actual_direction = "Neutral"
        if morning_price > 0:
            pct_change = (actual_close - morning_price) / morning_price
            if pct_change >  0.01: actual_direction = "Bullish"
            elif pct_change < -0.01: actual_direction = "Bearish"

        why_wrong = ""
        if morning_pred != actual_direction:
            why_wrong = (
                f"Predicted {morning_pred} but actual was {actual_direction}. "
                f"Price moved {((actual_close-morning_price)/morning_price*100):.2f}%."
            )

        try:
            rag_agent("log_evaluation", {
                "stock":            company,
                "actual_close":     actual_close,
                "actual_direction": actual_direction,
                "why_wrong":        why_wrong,
            })
        except Exception as e:
            logger.error(f"  {company}: RAG log error: {e}")

        correct = morning_pred == actual_direction
        logger.info(
            f"  {company.upper():12} → "
            f"Pred: {morning_pred:8} | "
            f"Actual: {actual_direction:8} | "
            f"{'✅' if correct else '❌'}"
        )

    try:
        rag_agent("archive_daily", {})
    except Exception as e:
        logger.error(f"  Archive error: {e}")


async def run_retraining() -> None:
    """
    End-of-day retraining at 4:30 PM.
    FIX: uses TRAINING_DATA_DIR from settings, not hardcoded path.
    FIX: train_model properly imported from prediction_agent.
    """
    logger.info(f"\n{'='*60}\n END OF DAY RETRAINING (4:30 PM)\n{'='*60}")

    today = str(date.today())

    for company, symbol in STOCK_CONFIG.items():
        # Check if already retrained today
        try:
            check = rag_agent("check_trained", {"stock": company, "date": today})
            if check.get("already_trained"):
                logger.info(f"  {company}: already retrained today — skipping")
                continue
        except Exception:
            pass

        # Load today's evaluated data
        try:
            global_df  = pd.read_csv(GLOBAL_REPORT_PATH)
            today_rows = global_df[
                (global_df["stock"] == company) &
                (global_df["date"]  == today)   &
                (global_df["evaluated"] == True)
            ]
        except Exception as e:
            logger.error(f"  {company}: global_report read error: {e}")
            continue

        if today_rows.empty:
            logger.warning(f"  {company}: no evaluated data for today")
            continue

        # FIX: use TRAINING_DATA_DIR from settings (not hardcoded path)
        training_path = os.path.join(TRAINING_DATA_DIR, f"{company}.csv")
        if not os.path.exists(training_path):
            logger.warning(f"  {company}: training CSV not found at {training_path}")
            continue

        training_df = pd.read_csv(training_path)
        new_rows    = today_rows[~today_rows["date"].isin(training_df["date"])]

        if new_rows.empty:
            logger.info(f"  {company}: today's data already in training CSV")
            continue

        training_df = pd.concat(
            [training_df, new_rows[training_df.columns]],
            ignore_index=True
        )
        training_df.to_csv(training_path, index=False)

        logger.info(
            f"  {company}: retraining on "
            f"{len(training_df)} rows (+{len(new_rows)} new)..."
        )

        try:
            # FIX: train_model imported from prediction_agent
            metadata = train_model(company, training_path)
            rag_agent("log_retraining", {
                "stock":                 company,
                "symbol":                symbol,
                "training_rows_used":    len(training_df),
                "date_range_start":      str(training_df["date"].min()),
                "date_range_end":        str(training_df["date"].max()),
                "walk_forward_accuracy": metadata["walk_forward_accuracy"],
                "new_rows_added":        len(new_rows),
            })
            logger.info(
                f"  ✅ {company}: retrained | "
                f"acc={metadata['walk_forward_accuracy']:.1f}%"
            )
        except Exception as e:
            logger.error(f"  {company}: retraining failed: {e}")

    logger.info(f"{'='*60}\n RETRAINING COMPLETE\n{'='*60}")