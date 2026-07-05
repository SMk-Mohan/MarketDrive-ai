"""
agents/rag_agent.py
RAG Agent — semantic search, prediction logging,
evaluation tracking, API usage management, and LLM explanation.
Uses FAISS + all-MiniLM-L6-v2 embeddings, one index per stock.
"""

import os
import json
import pickle
import asyncio
import logging
from datetime import datetime, date
from functools import partial

import numpy as np
import pandas as pd
import faiss
from groq import Groq
from sentence_transformers import SentenceTransformer

from config.settings import (
    RAG_DIR,
    GROQ_API_KEY,
    STOCK_CONFIG,
    API_LIMITS,
    MARKET_TIME_REPORT_PATH,
    GLOBAL_REPORT_PATH,
)

logger = logging.getLogger(__name__)

# ── Derived paths (all under RAG_DIR from settings) ──────────
FAISS_DIR            = os.path.join(RAG_DIR, "faiss_indexes")
EMBEDDER_DIR         = os.path.join(RAG_DIR, "embedder")
TRAINING_TRACKER_PATH = os.path.join(RAG_DIR, "training_data_tracker.csv")
API_TRACKER_PATH     = os.path.join(RAG_DIR, "api_tracker.json")

for _d in [RAG_DIR, FAISS_DIR, EMBEDDER_DIR]:
    os.makedirs(_d, exist_ok=True)

EMBEDDING_DIM = 384   # fixed for all-MiniLM-L6-v2

# ── CSV column schemas ────────────────────────────────────────
MARKET_TIME_REPORT_COLS = [
    "date", "time", "stock", "symbol",
    "morning_prediction", "final_prediction",
    "confidence", "price_range_low", "price_range_high",
    "rsi_at_pred", "macd_signal_at_pred",
    "volume_ratio_at_pred", "trend_score_at_pred",
    "nifty_trend_at_pred", "banknifty_trend_at_pred",
    "sensex_trend_at_pred", "sentiment_score_at_pred",
    "dominant_event_at_pred", "trigger_count", "trigger_reason",
    "actual_close", "actual_direction",
    "correct", "why_wrong", "evaluated",
]

GLOBAL_REPORT_COLS    = MARKET_TIME_REPORT_COLS + ["archived_at"]

TRAINING_TRACKER_COLS = [
    "stock", "symbol", "retrain_date",
    "training_rows_used", "date_range_start", "date_range_end",
    "walk_forward_accuracy", "model_version",
    "new_rows_added_today", "duplicate_check_passed",
]

API_TRACKER_DEFAULTS = {
    "date": str(date.today()),
    "marketaux": {"calls_today": 0, "limit": API_LIMITS["marketaux"], "last_call": None},
    "groq":      {"calls_today": 0, "limit": API_LIMITS["groq"],      "last_call": None},
    "gdelt":     {"calls_today": 0, "limit": API_LIMITS["gdelt"],     "last_call": None},
}


# ══════════════════════════════════════════════════════════════
# LAZY SINGLETONS
# FIX: not loaded at module level — loaded on first use only
# ══════════════════════════════════════════════════════════════

_embedder: SentenceTransformer | None = None
_groq_client: Groq | None = None


def _get_embedder() -> SentenceTransformer:
    """Returns cached embedder, loads on first call."""
    global _embedder
    if _embedder is None:
        logger.info("[RAG Agent] Loading sentence embedder...")
        _embedder = SentenceTransformer(
            "all-MiniLM-L6-v2",
            cache_folder=EMBEDDER_DIR,
        )
        logger.info("[RAG Agent] Embedder ready")
    return _embedder


def _get_groq() -> Groq:
    """Returns cached Groq client, creates on first call."""
    global _groq_client
    if _groq_client is None:
        _groq_client = Groq(api_key=GROQ_API_KEY)
    return _groq_client


# ══════════════════════════════════════════════════════════════
# STORAGE INITIALIZER
# FIX: called explicitly from events.startup(), not at import
# ══════════════════════════════════════════════════════════════

def init_storage():
    """
    Creates all storage files with correct headers if missing.
    Call once from core/events.py startup(), not at import time.
    """
    if not os.path.exists(MARKET_TIME_REPORT_PATH):
        pd.DataFrame(columns=MARKET_TIME_REPORT_COLS).to_csv(
            MARKET_TIME_REPORT_PATH, index=False
        )
        logger.info("[RAG Agent] Created market_time_report.csv")

    if not os.path.exists(GLOBAL_REPORT_PATH):
        pd.DataFrame(columns=GLOBAL_REPORT_COLS).to_csv(
            GLOBAL_REPORT_PATH, index=False
        )
        logger.info("[RAG Agent] Created global_report.csv")

    if not os.path.exists(TRAINING_TRACKER_PATH):
        pd.DataFrame(columns=TRAINING_TRACKER_COLS).to_csv(
            TRAINING_TRACKER_PATH, index=False
        )
        logger.info("[RAG Agent] Created training_data_tracker.csv")

    if not os.path.exists(API_TRACKER_PATH):
        with open(API_TRACKER_PATH, "w") as f:
            json.dump(API_TRACKER_DEFAULTS, f, indent=2)
        logger.info("[RAG Agent] Created api_tracker.json")

    logger.info("[RAG Agent] Storage initialized")


# ══════════════════════════════════════════════════════════════
# FAISS INDEX MANAGER
# ══════════════════════════════════════════════════════════════

def _faiss_paths(stock: str) -> tuple[str, str]:
    return (
        os.path.join(FAISS_DIR, f"{stock}_faiss.index"),
        os.path.join(FAISS_DIR, f"{stock}_rows.pkl"),
    )


def _load_index(stock: str) -> tuple:
    """Loads FAISS index + row list for a stock. Creates empty if missing."""
    idx_path, rows_path = _faiss_paths(stock)
    if os.path.exists(idx_path) and os.path.exists(rows_path):
        index = faiss.read_index(idx_path)
        with open(rows_path, "rb") as f:
            rows = pickle.load(f)
    else:
        index = faiss.IndexFlatL2(EMBEDDING_DIM)
        rows  = []
    return index, rows


def _save_index(stock: str, index, rows: list):
    idx_path, rows_path = _faiss_paths(stock)
    faiss.write_index(index, idx_path)
    with open(rows_path, "wb") as f:
        pickle.dump(rows, f)


def _row_to_text(row: dict) -> str:
    """
    Converts a market_time_report row to embeddable text.
    This is the text that gets semantically searched.
    """
    correct_str = "correct" if row.get("correct") else "wrong"
    return (
        f"{row.get('stock','?')} on {row.get('date','?')}: "
        f"RSI {row.get('rsi_at_pred','?')}, "
        f"MACD {row.get('macd_signal_at_pred','?')}, "
        f"Nifty {row.get('nifty_trend_at_pred','?')}, "
        f"BankNifty {row.get('banknifty_trend_at_pred','?')}, "
        f"Sensex {row.get('sensex_trend_at_pred','?')}, "
        f"Sentiment {float(row.get('sentiment_score_at_pred', 0)):.2f}, "
        f"Volume ratio {float(row.get('volume_ratio_at_pred', 1)):.2f}, "
        f"Dominant event {row.get('dominant_event_at_pred','None')}, "
        f"Trigger {row.get('trigger_reason','None')}, "
        f"Prediction {row.get('final_prediction','?')} ({correct_str})"
    )


def _embed_and_add(stock: str, row: dict):
    """Embeds a completed evaluated row and adds to FAISS index."""
    index, rows = _load_index(stock)
    text        = _row_to_text(row)
    embedding   = _get_embedder().encode([text]).astype("float32")
    index.add(embedding)
    rows.append(row)
    _save_index(stock, index, rows)
    logger.info(f"[RAG Agent] Indexed {stock} row for {row.get('date')}")


# ══════════════════════════════════════════════════════════════
# API TRACKER
# ══════════════════════════════════════════════════════════════

def _load_api_tracker() -> dict:
    """Loads tracker JSON, resets counts if it's a new day."""
    with open(API_TRACKER_PATH, "r") as f:
        tracker = json.load(f)

    today = str(date.today())
    if tracker.get("date") != today:
        for svc in ["marketaux", "groq", "gdelt"]:
            tracker[svc]["calls_today"] = 0
            tracker[svc]["last_call"]   = None
        tracker["date"] = today
        _save_api_tracker(tracker)

    return tracker


def _save_api_tracker(tracker: dict):
    with open(API_TRACKER_PATH, "w") as f:
        json.dump(tracker, f, indent=2)


def _log_api_call(service: str, calls: int = 1):
    tracker = _load_api_tracker()
    tracker[service]["calls_today"] += calls
    tracker[service]["last_call"]    = datetime.now().isoformat()
    _save_api_tracker(tracker)


def _get_api_remaining(service: str) -> int:
    tracker = _load_api_tracker()
    limit   = API_LIMITS.get(service, 0)
    used    = tracker.get(service, {}).get("calls_today", 0)
    return max(0, limit - used)


def _get_api_status() -> dict:
    tracker = _load_api_tracker()
    return {
        svc: {
            "used":      tracker[svc]["calls_today"],
            "limit":     API_LIMITS[svc],
            "remaining": max(0, API_LIMITS[svc] - tracker[svc]["calls_today"]),
            "last_call": tracker[svc]["last_call"],
            "safe":      (API_LIMITS[svc] - tracker[svc]["calls_today"]) > 5,
        }
        for svc in ["marketaux", "groq", "gdelt"]
    }


# ══════════════════════════════════════════════════════════════
# PREDICTION LOGGER
# ══════════════════════════════════════════════════════════════

def _log_prediction(
    stock:          str,
    symbol:         str,
    prediction:     str,
    confidence:     float,
    price_range_low:  float,   # FIX: param name matches coordinator key
    price_range_high: float,   # FIX: param name matches coordinator key
    features:       dict,
    trigger_count:  int  = 0,
    trigger_reason: str  = "None",
    is_morning:     bool = True,
) -> str:
    """
    Logs prediction to market_time_report.csv.
    Morning call: appends new row.
    Re-prediction call: updates existing row for today.
    """
    now    = datetime.now()
    today  = now.strftime("%Y-%m-%d")
    row_id = f"{today}_{stock}"

    df       = pd.read_csv(MARKET_TIME_REPORT_PATH)
    existing = df[(df["date"] == today) & (df["stock"] == stock)]

    new_row = {
        "date":                    today,
        "time":                    now.strftime("%H:%M:%S"),
        "stock":                   stock,
        "symbol":                  symbol,
        "morning_prediction":      prediction if is_morning
                                   else (existing["morning_prediction"].values[0]
                                         if len(existing) > 0 else prediction),
        "final_prediction":        prediction,
        "confidence":              confidence,
        "price_range_low":         price_range_low,
        "price_range_high":        price_range_high,
        "rsi_at_pred":             features.get("rsi",              50.0),
        "macd_signal_at_pred":     features.get("macd_signal",      0.0),
        "volume_ratio_at_pred":    features.get("volume_ratio",     1.0),
        "trend_score_at_pred":     features.get("trend_score",      0),
        "nifty_trend_at_pred":     features.get("nifty_trend",      "Neutral"),
        "banknifty_trend_at_pred": features.get("banknifty_trend",  "Neutral"),
        "sensex_trend_at_pred":    features.get("sensex_trend",     "Neutral"),
        "sentiment_score_at_pred": features.get("sentiment_score",  0.0),
        "dominant_event_at_pred":  features.get("dominant_event",   "Other"),
        "trigger_count":           trigger_count,
        "trigger_reason":          trigger_reason,
        "actual_close":            None,
        "actual_direction":        None,
        "correct":                 None,
        "why_wrong":               None,
        "evaluated":               False,
    }

    if len(existing) > 0 and not is_morning:
        # FIX: update per column, not fragile list assignment
        for col, val in new_row.items():
            df.loc[
                (df["date"] == today) & (df["stock"] == stock), col
            ] = val
    else:
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    df.to_csv(MARKET_TIME_REPORT_PATH, index=False)
    logger.info(
        f"[RAG Agent] Logged: {stock} → {prediction} "
        f"({confidence:.1f}%) trigger={trigger_reason}"
    )
    return row_id


# ══════════════════════════════════════════════════════════════
# EVALUATION LOGGER
# ══════════════════════════════════════════════════════════════

def _log_evaluation(
    stock:            str,
    actual_close:     float,
    actual_direction: str,
    why_wrong:        str = "",
):
    """
    Fills evaluation columns for today's prediction at 3:30 PM.
    Then embeds the completed row into FAISS for future RAG search.
    """
    today = datetime.now().strftime("%Y-%m-%d")
    df    = pd.read_csv(MARKET_TIME_REPORT_PATH)
    mask  = (df["date"] == today) & (df["stock"] == stock)

    if mask.sum() == 0:
        logger.warning(f"[RAG Agent] No prediction for {stock} on {today}")
        return

    final_pred = df.loc[mask, "final_prediction"].values[0]
    correct    = (final_pred == actual_direction)

    df.loc[mask, "actual_close"]     = actual_close
    df.loc[mask, "actual_direction"] = actual_direction
    df.loc[mask, "correct"]          = correct
    df.loc[mask, "why_wrong"]        = why_wrong if not correct else ""
    df.loc[mask, "evaluated"]        = True
    df.to_csv(MARKET_TIME_REPORT_PATH, index=False)

    # Embed completed row into FAISS
    _embed_and_add(stock, df[mask].iloc[0].to_dict())

    logger.info(
        f"[RAG Agent] Evaluated {stock}: "
        f"pred={final_pred} actual={actual_direction} "
        f"→ {'✅' if correct else '❌'}"
    )


# ══════════════════════════════════════════════════════════════
# ARCHIVE
# ══════════════════════════════════════════════════════════════

def _archive_daily_report():
    """
    Moves evaluated rows from market_time_report → global_report.
    Clears only evaluated rows from daily report.
    """
    daily_df  = pd.read_csv(MARKET_TIME_REPORT_PATH)
    global_df = pd.read_csv(GLOBAL_REPORT_PATH)

    evaluated = daily_df[daily_df["evaluated"] == True].copy()
    if len(evaluated) == 0:
        logger.info("[RAG Agent] No evaluated rows to archive")
        return

    evaluated["archived_at"] = datetime.now().isoformat()
    global_df = pd.concat([global_df, evaluated], ignore_index=True)
    global_df.to_csv(GLOBAL_REPORT_PATH, index=False)

    remaining = daily_df[daily_df["evaluated"] != True]
    remaining.to_csv(MARKET_TIME_REPORT_PATH, index=False)

    logger.info(
        f"[RAG Agent] Archived {len(evaluated)} rows | "
        f"{len(remaining)} unevaluated remain"
    )


# ══════════════════════════════════════════════════════════════
# TRAINING TRACKER
# ══════════════════════════════════════════════════════════════

def _is_already_trained(stock: str, target_date: str) -> bool:
    if not os.path.exists(TRAINING_TRACKER_PATH):
        return False
    df = pd.read_csv(TRAINING_TRACKER_PATH)
    if df.empty:
        return False
    return len(df[(df["stock"] == stock) & (df["retrain_date"] == target_date)]) > 0


def _log_retraining(
    stock:                 str,
    symbol:                str,
    training_rows_used:    int,
    date_range_start:      str,
    date_range_end:        str,
    walk_forward_accuracy: float,
    new_rows_added:        int,
    # FIX: model_version is optional with default — coordinator doesn't always pass it
    model_version:         str = "",
):
    df = pd.read_csv(TRAINING_TRACKER_PATH)
    today = str(date.today())
    if not model_version:
        model_version = f"{stock}_model_v{today}.pkl"

    new_row = {
        "stock":                  stock,
        "symbol":                 symbol,
        "retrain_date":           today,
        "training_rows_used":     training_rows_used,
        "date_range_start":       date_range_start,
        "date_range_end":         date_range_end,
        "walk_forward_accuracy":  walk_forward_accuracy,
        "model_version":          model_version,
        "new_rows_added_today":   new_rows_added,
        "duplicate_check_passed": True,
    }
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(TRAINING_TRACKER_PATH, index=False)
    logger.info(
        f"[RAG Agent] Retraining logged: {stock} | "
        f"acc={walk_forward_accuracy:.1f}% | "
        f"rows={training_rows_used} | new={new_rows_added}"
    )


# ══════════════════════════════════════════════════════════════
# SEMANTIC SEARCH
# ══════════════════════════════════════════════════════════════

def _search_similar(stock: str, query_row: dict, top_k: int = 3) -> list:
    """
    Finds top_k most semantically similar past situations
    for a given stock using FAISS vector search.
    """
    index, rows = _load_index(stock)
    if index.ntotal == 0:
        logger.info(f"[RAG Agent] No past situations indexed for {stock} yet")
        return []

    query_text = _row_to_text(query_row)
    query_vec  = _get_embedder().encode([query_text]).astype("float32")
    k          = min(top_k, index.ntotal)
    distances, indices = index.search(query_vec, k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < len(rows):
            row = rows[idx].copy()
            row["similarity"] = round(float(1 / (1 + dist)), 4)
            results.append(row)
    return results


# ══════════════════════════════════════════════════════════════
# EXPLANATION GENERATOR
# ══════════════════════════════════════════════════════════════

def _explain_prediction(
    stock:          str,
    today_features: dict,
    prediction:     str,
    confidence:     float,
    shap_top:       list,
) -> str:
    """
    Full RAG explanation pipeline:
    1. Semantic search for similar past situations
    2. Build context string
    3. Call Groq/Llama for natural language explanation
    Returns 3-4 sentence explanation for dashboard.
    """
    similar = _search_similar(stock, today_features, top_k=3)

    past_context = (
        "\n".join([
            f"- {s.get('date')}: "
            f"RSI {s.get('rsi_at_pred')}, "
            f"Nifty {s.get('nifty_trend_at_pred')}, "
            f"Sentiment {float(s.get('sentiment_score_at_pred', 0)):.2f}, "
            f"predicted {s.get('final_prediction')} "
            f"({'✓ correct' if s.get('correct') else '✗ wrong'})"
            for s in similar
        ])
        if similar else "No similar past situations indexed yet."
    )

    shap_context = "\n".join([
        f"- {f['feature']}: {f['shap_value']:+.4f} "
        f"({'supporting' if f['shap_value'] > 0 else 'opposing'} {prediction})"
        for f in shap_top[:4]
    ]) if shap_top else "No SHAP data available."

    prompt = f"""You are a financial analyst explaining a stock prediction.

Stock: {stock.upper()}
Today's Prediction: {prediction} ({confidence:.1f}% confidence)

Key Features Driving This Prediction (SHAP values):
{shap_context}

Similar Past Situations For This Stock:
{past_context}

Today's Market Context:
- RSI: {today_features.get('rsi', '?')}
- Nifty Trend: {today_features.get('nifty_trend', '?')}
- BankNifty Trend: {today_features.get('banknifty_trend', '?')}
- Volume Ratio: {today_features.get('volume_ratio', '?')}
- Sentiment Score: {today_features.get('sentiment_score', '?')}
- Dominant Event: {today_features.get('dominant_event', '?')}

Write a clear, 3-4 sentence explanation of WHY this stock is predicted {prediction} today.
Mention key technical factors, news sentiment if relevant, and what similar past situations suggest.
Be specific. Do not use bullet points."""

    try:
        response = _get_groq().chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=250,
            timeout=30.0,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"[RAG Agent] CRITICAL: Groq reasoning failed. Error: {str(e)}")
        # Provide a more detailed technical fallback than the one in coordinator
        return (
            f"Technical Analysis: {stock.upper()} shows {prediction} strength. "
            f"RSI is at {today_features.get('rsi', '?')} with Nifty trending {today_features.get('nifty_trend', '?')}. "
            f"Sentiment is {float(today_features.get('sentiment_score', 0)):.2f} based on recent high-impact events."
        )


# ══════════════════════════════════════════════════════════════
# MASTER DISPATCHER (sync)
# ══════════════════════════════════════════════════════════════

def rag_agent(action: str, payload: dict) -> dict:
    """
    Single entry point for all RAG operations.
    Coordinator calls this with action + payload.

    Actions:
        log_prediction  → log morning / re-prediction
        log_evaluation  → fill actual result at 3:30 PM
        archive_daily   → move daily → global at 4:00 PM
        log_retraining  → record retraining event
        check_trained   → has this date already been retrained?
        explain         → generate RAG + LLM explanation
        api_status      → full API usage summary
        api_remaining   → remaining calls for one service
        log_api_call    → increment API counter
        search_similar  → raw FAISS search (no LLM)
    """
    action = action.lower().strip()

    try:
        if action == "log_prediction":
            row_id = _log_prediction(
                stock            = payload["stock"],
                symbol           = payload["symbol"],
                prediction       = payload["prediction"],
                confidence       = payload["confidence"],
                price_range_low  = payload["price_range_low"],   # FIX: correct key
                price_range_high = payload["price_range_high"],  # FIX: correct key
                features         = payload["features"],
                trigger_count    = payload.get("trigger_count",  0),
                trigger_reason   = payload.get("trigger_reason", "None"),
                is_morning       = payload.get("is_morning",     True),
            )
            return {"status": "ok", "row_id": row_id}

        elif action == "log_evaluation":
            _log_evaluation(
                stock            = payload["stock"],
                actual_close     = payload["actual_close"],
                actual_direction = payload["actual_direction"],
                why_wrong        = payload.get("why_wrong", ""),
            )
            return {"status": "ok"}

        elif action == "archive_daily":
            _archive_daily_report()
            return {"status": "ok"}

        elif action == "log_retraining":
            _log_retraining(
                stock                 = payload["stock"],
                symbol                = payload["symbol"],
                training_rows_used    = payload["training_rows_used"],
                date_range_start      = payload["date_range_start"],
                date_range_end        = payload["date_range_end"],
                walk_forward_accuracy = payload["walk_forward_accuracy"],
                new_rows_added        = payload["new_rows_added"],
                # FIX: model_version optional — defaults to auto-generated name
                model_version         = payload.get("model_version", ""),
            )
            return {"status": "ok"}

        elif action == "check_trained":
            return {
                "status":          "ok",
                "already_trained": _is_already_trained(
                    payload["stock"], payload["date"]
                ),
            }

        elif action == "explain":
            explanation = _explain_prediction(
                stock           = payload["stock"],
                today_features  = payload["features"],
                prediction      = payload["prediction"],
                confidence      = payload["confidence"],
                shap_top        = payload.get("shap_top", []),
            )
            # FIX: log groq call here in dispatcher, not inside explain fn
            _log_api_call("groq", 1)
            return {"status": "ok", "explanation": explanation}

        elif action == "api_status":
            return {"status": "ok", "data": _get_api_status()}

        elif action == "api_remaining":
            return {
                "status":    "ok",
                "remaining": _get_api_remaining(payload["service"]),
            }

        elif action == "log_api_call":
            _log_api_call(
                payload["service"],
                payload.get("calls", 1),
            )
            return {"status": "ok"}

        elif action == "search_similar":
            results = _search_similar(
                stock     = payload["stock"],
                query_row = payload["features"],
                top_k     = payload.get("top_k", 3),
            )
            return {"status": "ok", "results": results}

        else:
            logger.error(f"[RAG Agent] Unknown action: {action}")
            return {"status": "error", "message": f"Unknown action: {action}"}

    except Exception as e:
        logger.error(f"[RAG Agent] Error in action={action}: {e}")
        return {"status": "error", "message": str(e)}


# ══════════════════════════════════════════════════════════════
# ASYNC WRAPPER — for coordinator nodes
# FIX: rag_agent is sync (disk/CSV I/O) — run in executor
# ══════════════════════════════════════════════════════════════

async def run_rag_agent(action: str, payload: dict) -> dict:
    """
    Async wrapper so coordinator nodes don't block the event loop.
    Use this when calling from async context (FastAPI, coordinator).
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None, partial(rag_agent, action, payload)
    )