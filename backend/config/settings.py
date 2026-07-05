import os
import sys
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# API KEYS — validated on import, hard fail if missing
# ============================================================

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
MARKETAUX_API_KEY = os.getenv("MARKETAUX_API_KEY", "")

if not GROQ_API_KEY:
    sys.exit("[FATAL] GROQ_API_KEY is not set in .env — aborting.")
if not MARKETAUX_API_KEY:
    sys.exit("[FATAL] MARKETAUX_API_KEY is not set in .env — aborting.")

# ============================================================
# PROJECT ROOT
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ============================================================
# DIRECTORIES — auto-created on startup
# ============================================================

MODEL_DIR         = os.path.join(BASE_DIR, "models")
ENCODER_DIR       = os.path.join(BASE_DIR, "encoders")
TRAINING_DATA_DIR = os.path.join(BASE_DIR, "training_data")
RAG_DIR           = os.path.join(BASE_DIR, "rag")
CACHE_DIR         = os.path.join(BASE_DIR, "cache")

for _dir in [MODEL_DIR, ENCODER_DIR, TRAINING_DATA_DIR, RAG_DIR, CACHE_DIR]:
    os.makedirs(_dir, exist_ok=True)

# ============================================================
# RAG FILE PATHS
# ============================================================

MARKET_TIME_REPORT_PATH = os.path.join(RAG_DIR, "market_time_report.csv")
GLOBAL_REPORT_PATH      = os.path.join(RAG_DIR, "global_report.csv")
API_USAGE_PATH          = os.path.join(RAG_DIR, "api_usage.json")

# ============================================================
# STOCK CONFIGURATION
# NOTE: TMPV.NS (Tata Motors PV post-demerger) may not be
#       available on yfinance yet. TATAMOTORS.NS is used as
#       the data source; display name remains TMPV.
# ============================================================

STOCK_CONFIG = {
    "infosys":  "INFY.NS",
    "vodafone": "IDEA.NS",
    "tata":     "TMCV.NS",   # Corrected ticker
    "adani":    "ADANIENT.NS",
    "yesbank":  "YESBANK.NS",
}

# Display ticker override (for API responses / frontend)
STOCK_DISPLAY_TICKER = {
    "INFY.NS":        "INFY",
    "IDEA.NS":        "IDEA",
    "TMCV.NS":        "TMPV",      # show as TMPV in UI
    "ADANIENT.NS":    "ADANIENT",
    "YESBANK.NS":     "YESBANK",
}

# ============================================================
# SECTOR INDEX MAPPING
# ============================================================

SECTOR_INDEX_MAP = {
    "INFY.NS":       "^CNXIT",
    "IDEA.NS":       "^CNXIT",
    "TMCV.NS":       "^CNXAUTO",
    "ADANIENT.NS":   "^NSEI",
    "YESBANK.NS":    "^NSEBANK",
}

# ============================================================
# MACRO INDICES
# ============================================================

INDEX_SYMBOLS = {
    "nifty":     "^NSEI",
    "banknifty": "^NSEBANK",
    "sensex":    "^BSESN",
}

# ============================================================
# MODEL FEATURE SCHEMA
# MUST MATCH TRAINING DATA EXACTLY — do not reorder
#
# trend_score    : computed by market_agent (0–5 bull score)
# price_vs_ema20 : computed by market_agent (close - ema_20)
# *_enc columns  : encoded by prediction_agent from categoricals
# ============================================================

FEATURE_COLUMNS = [
    # Price (Only 'open' is known at 9:15 AM)
    "open",

    # Oscillators (Calculated on T-1 basis in production)
    "rsi", "macd", "macd_signal", "macd_histogram",

    # Trend
    "ema_20", "ema_50", "ema_200",

    # Volatility
    "atr", "atr_pct",

    # Volume
    "volume_ratio",

    # Derived trend features
    "trend_score",
    "price_vs_ema20",

    # Macro index encoded categoricals
    "nifty_trend_enc",
    "banknifty_trend_enc",
    "sensex_trend_enc",

    # News features
    "sentiment_score",
    "impact_score",
    "relevance_score",
    "news_count",

    # Dominant event encoded categorical
    "dominant_event_enc",

    # Re-prediction trigger count
    "trigger_count",
]

# Raw categorical columns before encoding
CATEGORICAL_COLUMNS = [
    "nifty_trend",
    "banknifty_trend",
    "sensex_trend",
    "dominant_event",
]

# ============================================================
# API LIMITS
# ============================================================

API_LIMITS = {
    "marketaux": 100,
    "groq":      14400,
    "gdelt":     999999,
}

# ============================================================
# TRIGGER THRESHOLDS (event-based re-prediction)
# ============================================================

VOLUME_SPIKE_TRIGGER  = 2.5   # volume_ratio threshold
NIFTY_MOVE_TRIGGER    = 1.0   # % move in Nifty
NEWS_IMPACT_TRIGGER   = 0.70  # impact_score threshold

# ============================================================
# MARKET HOURS (IST)
# ============================================================

MARKET_OPEN       = "09:15"
MARKET_CLOSE      = "15:30"
PREDICTION_TIME   = "09:15"
PREDICTION_CUTOFF = "15:15"
EVALUATION_TIME   = "15:30"
CLEANUP_TIME      = "16:30"
ARCHIVE_TIME      = "16:00"
RETRAIN_TIME      = "16:30"

# ============================================================
# TRUSTED NEWS SOURCES
# ============================================================

GOOD_SOURCES = [
    "reuters.com",
    "economictimes.indiatimes.com",
    "moneycontrol.com",
    "financialexpress.com",
]

# ============================================================
# CACHE TTL (seconds)
# ============================================================

NEWS_CACHE_TTL   = 900    # 15 min
MARKET_CACHE_TTL = 300    # 5 min

# ============================================================
# RISK THRESHOLDS
# ============================================================

HIGH_VOLATILITY_ATR_PCT      = 3.0
LOW_CONFIDENCE_THRESHOLD     = 55
MEDIUM_CONFIDENCE_THRESHOLD  = 65

# ============================================================
# MODEL SETTINGS
# ============================================================

MIN_TRAINING_ROWS = 250
RANDOM_STATE      = 42
TEST_SIZE         = 0.20

# ============================================================
# APP METADATA
# ============================================================

PROJECT_NAME = "MarketDrive AI"
VERSION      = "1.0.0"
TIMEZONE     = "Asia/Kolkata"