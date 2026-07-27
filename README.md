# MarketDrive AI
> **Multi-Agent Financial Intelligence & Short-Term Equity Forecasting System**

MarketDrive AI is an autonomous, multi-agent financial research system built for NSE (Indian Stock Market) top equities. It orchestrates 6 specialized AI agents using **LangGraph** to process technical market data, institutional news sentiment, machine learning feature importances, and risk management guidelines into actionable, explainable short-term predictions.

Live Demo → [market-drive-ai.vercel.app](https://market-drive-ai.vercel.app) · API → [marketdrive-backend.onrender.com](https://marketdrive-backend.onrender.com)

---

## System Architecture & Agent Flow

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                     MarketDrive AI — Multi-Agent Pipeline                    │
└──────────────────────────────────────────────────────────────────────────────┘

 User Request / Scheduled Cron (09:15 AM IST)
             │
             ▼
  ┌─────────────────────┐
  │  LangGraph           │  ← Stateful orchestration graph with conditional routing
  │  Coordinator         │
  └────────┬────────────┘
           │
     ┌─────┴──────────────────────────────────────────────────┐
     │                                                        │
     ▼                                                        │
┌─────────────┐   structured JSON    ┌──────────────────────┐ │
│  1. News     │──────────────────────▶  2. Market Agent     │ │
│  Agent       │  (LLM sentiment,     │  (OHLCV + Technicals)│ │
│              │   event tags,        └──────────┬───────────┘ │
│  MarketAux   │   impact scores)                │             │
│  + Groq LLM  │                                 ▼             │
└─────────────┘                     ┌─────────────────────┐   │
                                    │  3. Prediction Agent │   │
                                    │  XGBoost + SHAP      │   │
                                    │  Walk-Forward CV     │   │
                                    └──────────┬──────────┘   │
                                               │              │
                                               ▼              │
                                    ┌─────────────────────┐   │
                                    │  4. Coordinator Node │◀──┘
                                    │  Conflict Detection  │
                                    │  Confidence Calibration│
                                    └──────────┬──────────┘
                                               │
                             ┌─────────────────┴──────────────────┐
                             │                                    │
                             ▼                                    ▼
                  ┌─────────────────┐                 ┌──────────────────┐
                  │  5. Risk Agent  │                 │  6. RAG Agent    │
                  │  ATR Corridors  │                 │  FAISS + Groq    │
                  │  Volatility     │                 │  Analyst Prose   │
                  │  Regime         │                 │  + Logging       │
                  └────────┬────────┘                 └────────┬─────────┘
                           │                                   │
                           └─────────────────┬─────────────────┘
                                             ▼
                                  ┌─────────────────────┐
                                  │  FastAPI REST Layer  │
                                  │  + MongoDB Cache     │
                                  └──────────┬──────────┘
                                             ▼
                                  ┌─────────────────────┐
                                  │  React + Vite UI     │
                                  │  Vercel Deployment   │
                                  └─────────────────────┘
```

---

## Multi-Agent Roles

### 1. News Agent
- Fetches live financial headlines from **MarketAux API** with relevance and recency filtering.
- Applies exponential time-decay freshness scoring.
- Uses **Groq (Llama 3)** to extract structured sentiment scores (−1.0 to +1.0), event impact scores, dominant event classification, and narrative consistency checks.

### 2. Market Agent
- Ingests 12-month historical OHLCV data from **Yahoo Finance (yfinance)**.
- Computes a comprehensive technical indicator stack: **RSI, MACD, EMA 20/50/200, ATR, Volume Ratio, VWAP**.
- Evaluates macro market trends (Nifty, BankNifty, Sensex) and detects volume spikes and Support/Resistance breakouts.

### 3. Prediction Agent
- Runs live inference using per-stock **XGBoost classifiers** trained with walk-forward cross-validation.
- Dynamically labels training targets based on intraday returns `(Close − Open) / Open` with a balanced 1% threshold.
- Generates **SHAP feature importances** for every prediction, providing full model interpretability.

### 4. Coordinator Node
- Detects cross-signal conflicts (e.g. negative news vs. bullish technicals).
- Dynamically adjusts model confidence based on signal agreement (clamped 10%–95%).
- Monitors mid-session event triggers to initiate re-prediction sweeps automatically.

### 5. Risk Agent
- Classifies the current volatility regime (Low / Medium / High) based on ATR percentile.
- Generates ATR-based stop-loss and target price corridors, and provides position sizing guidance.

### 6. RAG Agent (Retrieval-Augmented Generation)
- Stores prediction history and semantic embeddings using **FAISS** (in-memory vector DB).
- Generates human-readable analyst-style explanations using **Groq LLM** via RAG prompts.
- Manages daily model retraining log storage and API rate limit tracking.

---

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| Orchestration | LangGraph | Stateful multi-agent graph with conditional routing |
| LLM Engine | Groq (Llama 3) | Sub-second inference for sentiment & RAG explanations |
| ML Model | XGBoost + SHAP | Intraday direction prediction with walk-forward CV |
| Market Data | yfinance + pandas_ta | OHLCV ingestion and technical indicator computation |
| News API | MarketAux | Live financial headline fetching |
| State Layer | MongoDB Atlas | Persistent cloud state sync for ephemeral hosts |
| Backend API | FastAPI + Uvicorn | Async REST server and scheduled prediction orchestration |
| Frontend UI | React 19 + Vite | Minimalist high-contrast analytics dashboard |
| Hosting | Render + Vercel | Backend and frontend cloud deployment |

---

## Key Design Decisions

### Why Groq?

Financial news sentiment processing must complete before market open (09:15 AM IST). Groq's LPU hardware delivers Llama 3 completions in under 200ms per call — fast enough to process 5 stocks × multiple headlines in parallel without blocking the LangGraph pipeline. It also enforces strict JSON output schemas, eliminating fragile regex parsing of LLM responses.

### Why MongoDB for State?

Cloud deployment platforms like Render run on **ephemeral file systems** — every service restart wipes local disk state, including:

- Trained XGBoost model files (`.pkl`)
- FAISS vector index files
- JSON prediction caches
- Retraining logs

A custom sync layer (`db_sync.py`) serializes all local state files, encodes binary models in Base64, and syncs them bidirectionally with **MongoDB Atlas** on every startup and after every retraining cycle. This gives the backend persistent memory across restarts with no data loss.

---

## Local Development

### Backend

```bash
cd backend
python -m venv venv

# Activate (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Install
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Fill: GROQ_API_KEY, MARKETAUX_API_KEY, MONGODB_URI

# Run
uvicorn main:app --reload
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

---

## Environment Variables

```bash
# Backend (.env)
GROQ_API_KEY=
MARKETAUX_API_KEY=
MONGODB_URI=
```

---

## Deployment

| Service | Platform | Config |
|---|---|---|
| Backend API | Render (Python Web Service) | `render.yaml` |
| Frontend UI | Vercel (Static SPA) | Auto-detected Vite project |
| Scheduled Jobs | GitHub Actions | `.github/workflows/` |

GitHub Actions wakes up the Render backend during IST market hours (free tier spin-up mitigation) and triggers the daily prediction + retraining sweep automatically.

---

## Covered Stocks

| Company | NSE Ticker |
|---|---|
| Infosys | INFY.NS |
| Vodafone Idea | IDEA.NS |
| Tata Motors | TATAMOTORS.NS |
| Adani Enterprises | ADANIENT.NS |
| Yes Bank | YESBANK.NS |

---

*Built by [Mohan](https://www.linkedin.com/in/mohansuraboina2104)*
