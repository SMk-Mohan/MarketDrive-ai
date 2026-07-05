import asyncio
import logging
import requests
import json
from groq import Groq
from datetime import datetime, timedelta, timezone
from functools import partial

from config.settings import (
    MARKETAUX_API_KEY,
    GROQ_API_KEY,
    GOOD_SOURCES,
    NEWS_CACHE_TTL,
)

logger = logging.getLogger(__name__)

# Groq client — one instance at module level
_groq_client = Groq(api_key=GROQ_API_KEY)


# ── News Fetcher ──────────────────────────────────────────────

def get_company_news(company_name: str, days: int = 7,
                     limit: int = 10) -> list:
    """Fetches news from MarketAux for a given company name."""
    url        = "https://api.marketaux.com/v1/news/all"
    start_date = datetime.now(timezone.utc) - timedelta(days=days)
    params = {
        "api_token":       MARKETAUX_API_KEY,
        "search":          company_name,
        "language":        "en",
        "limit":           limit,
        "sort":            "published_desc",
        "published_after": start_date.strftime("%Y-%m-%dT%H:%M"),
    }
    try:
        response = requests.get(url, params=params, timeout=15)
        if response.status_code == 200:
            return response.json().get("data", [])
        logger.warning(f"[News Agent] MarketAux HTTP {response.status_code}")
        return []
    except Exception as e:
        logger.error(f"[News Agent] MarketAux fetch error: {e}")
        return []


# ── Article Extractor ─────────────────────────────────────────

def extract_article(url: str) -> str | None:
    """
    DEACTIVATED: Extracts full article text via newspaper3k.
    Disabled to prevent hangs on slow external websites.
    """
    return None


# ── Freshness Score ───────────────────────────────────────────

def calculate_freshness(published_at_str: str) -> float:
    """
    Returns freshness decay score:
    < 1h  → 1.0  |  < 6h → 0.8  |  < 24h → 0.6  |  older → 0.3
    """
    try:
        pub_date   = datetime.fromisoformat(
            published_at_str.replace("Z", "+00:00")
        )
        diff_hours = (
            datetime.now(timezone.utc) - pub_date
        ).total_seconds() / 3600

        if diff_hours < 1:  return 1.0
        if diff_hours < 6:  return 0.8
        if diff_hours < 24: return 0.6
        return 0.3
    except Exception:
        return 0.3


# ── Groq Sentiment Analyzer ───────────────────────────────────

def analyze_article(title: str, content: str,
                    historical_context: list) -> dict:
    """
    Sends article to Groq/Llama for structured sentiment analysis.

    Returns JSON with all fields the coordinator needs.
    Key field: dominant_event (matches CATEGORICAL_COLUMNS in settings).
    """
    prompt = f"""
You are a senior financial research agent analyzing stock market news.

Historical Trend Context (recent articles already processed):
{historical_context}

New Article Title: {title}
Article Content: {content[:4000]}

Return ONLY a valid JSON object with these exact keys:
{{
    "duplicate":            <bool>,
    "dominant_event":       <string — e.g. "Earnings", "Regulatory", "Macro", "Management", "Product", "Other">,
    "sentiment":            <string — one-line qualitative description>,
    "sentiment_shift":      <"Improvement" | "Deterioration" | "Stable">,
    "narrative_consistency":<"Strengthens" | "Weakens" | "Neutral">,
    "sentiment_score":      <float between -1.0 and 1.0>,
    "impact":               <"Low" | "Medium" | "High">,
    "impact_score":         <float between 0.0 and 1.0>,
    "relevance_score":      <float between 0.0 and 1.0>,
    "market_relevance":     <"Low" | "Medium" | "High">,
    "summary":              <string — one sentence max>
}}
"""
    try:
        response = _groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.1,
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        logger.error(f"[News Agent] Groq error: {e}")
        return _fallback_analysis()


def _fallback_analysis() -> dict:
    """Returns safe neutral defaults when Groq call fails."""
    return {
        "duplicate":             False,
        "dominant_event":        "Other",
        "sentiment":             "Neutral",
        "sentiment_shift":       "Stable",
        "narrative_consistency": "Neutral",
        "sentiment_score":       0.0,
        "impact":                "Low",
        "impact_score":          0.0,
        "relevance_score":       0.0,
        "market_relevance":      "Low",
        "summary":               "Analysis unavailable",
    }


# ── Sentiment Decay ───────────────────────────────────────────

def apply_decay(score: float, days_old: float,
                decay_factor: float = 0.85) -> float:
    """
    Applies exponential decay to a sentiment score.
    score * decay_factor^days_old
    Called by coordinator when carrying forward past sentiment.
    """
    return round(score * (decay_factor ** days_old), 4)


# ── Main News Agent (sync) ────────────────────────────────────

def news_agent_v3(company_name: str,
                  historical_summaries: list) -> dict:
    """
    Main News Agent. Sync function — call via run_in_executor
    from async coordinator to avoid blocking the event loop.

    Args:
        company_name        : e.g. "infosys"
        historical_summaries: list of past summaries for context

    Returns:
        dict with:
          articles      : list of processed article dicts
          news_count    : int — number of non-duplicate articles
          sentiment_score  : float — weighted avg sentiment
          impact_score     : float — avg impact score
          relevance_score  : float — avg relevance score
          dominant_event   : str   — most common event type
    """
    raw_news  = get_company_news(company_name)
    articles  = []

    for article in raw_news:
        # Filter to trusted sources only
        if not any(src in article.get("url", "")
                   for src in GOOD_SOURCES):
            continue

        content = (
            extract_article(article.get("url", ""))
            or article.get("description", "")
            or ""
        )

        analysis = analyze_article(
            article.get("title", ""),
            content,
            historical_summaries,
        )

        if analysis.get("duplicate", False):
            continue

        published_at = article.get("published_at", "")
        analysis.update({
            "headline":       article.get("title", ""),
            "url":            article.get("url", ""),
            "published_at":   published_at,
            "freshness_score": calculate_freshness(published_at),
        })

        articles.append(analysis)
        historical_summaries.append(analysis.get("summary", ""))

    # ── Aggregate features for coordinator ───────────────────
    news_count = len(articles)

    if news_count == 0:
        logger.warning(f"[News Agent] No articles found for {company_name}")
        return _empty_output()

    # Weighted by freshness + relevance
    weights = [
        a["freshness_score"] * a["relevance_score"]
        for a in articles
    ]
    total_w = sum(weights) or 1.0

    sentiment_score = round(
        sum(a["sentiment_score"] * w
            for a, w in zip(articles, weights)) / total_w, 4
    )
    impact_score = round(
        sum(a["impact_score"] * w
            for a, w in zip(articles, weights)) / total_w, 4
    )
    relevance_score = round(
        sum(a["relevance_score"] * w
            for a, w in zip(articles, weights)) / total_w, 4
    )

    # Most frequent dominant_event
    from collections import Counter
    event_counts   = Counter(a["dominant_event"] for a in articles)
    dominant_event = event_counts.most_common(1)[0][0]

    logger.info(
        f"[News Agent] {company_name}: {news_count} articles | "
        f"sentiment={sentiment_score:+.3f} | "
        f"dominant_event={dominant_event}"
    )

    return {
        "articles":       articles,
        "news_count":     news_count,
        "sentiment_score":  sentiment_score,
        "impact_score":     impact_score,
        "relevance_score":  relevance_score,
        "dominant_event":   dominant_event,
    }


def _empty_output() -> dict:
    return {
        "articles":        [],
        "news_count":      0,
        "sentiment_score": 0.0,
        "impact_score":    0.0,
        "relevance_score": 0.0,
        "dominant_event":  "Other",
    }


# ── Async wrapper for coordinator ─────────────────────────────

async def run_news_agent(company_name: str,
                          historical_summaries: list) -> dict:
    """
    Async wrapper — runs sync news_agent_v3 in thread pool
    so it doesn't block FastAPI's event loop.
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        partial(news_agent_v3, company_name, historical_summaries)
    )
