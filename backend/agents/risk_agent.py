"""
agents/risk_agent.py
Evaluates prediction trustworthiness using cross-agent signals.
Standalone — no external API calls, pure scoring logic.
"""

import logging
from config.settings import (
    HIGH_VOLATILITY_ATR_PCT,
    LOW_CONFIDENCE_THRESHOLD,
    MEDIUM_CONFIDENCE_THRESHOLD,
)

logger = logging.getLogger(__name__)


def risk_agent(payload: dict) -> dict:
    """
    Evaluates prediction risk from combined agent outputs.

    Args:
        payload: merged dict from News + Market + Prediction agents.
                 Required keys:
                   prediction, confidence, probabilities,
                   volatility_regime, volume_ratio, volume_spike,
                   market_breadth, sr_signal, sentiment_score,
                   impact_score, dominant_event,
                   nifty_trend, banknifty_trend, sensex_trend,
                   signals_conflicting, atr_pct

    Returns:
        risk dict with risk, risk_score, risk_factors,
        confidence_reliable, probability_margin, trade_suggestion
    """
    score   = 0.0
    factors = []

    company = payload.get("company", payload.get("stock", "unknown"))

    # ── 1. Volatility ─────────────────────────────────────────
    vol_regime = payload.get("volatility_regime", "Medium")
    if vol_regime == "High":
        score += 0.20
        factors.append("High ATR — volatile market conditions")
    elif vol_regime == "Medium":
        score += 0.10

    # ── 2. Model confidence ───────────────────────────────────
    confidence = payload.get("confidence", 50.0)
    if confidence < LOW_CONFIDENCE_THRESHOLD:
        score += 0.20
        factors.append(f"Low model confidence ({confidence:.1f}%)")
    elif confidence < MEDIUM_CONFIDENCE_THRESHOLD:
        score += 0.10
        factors.append(f"Moderate confidence ({confidence:.1f}%)")

    # ── 3. Probability margin ─────────────────────────────────
    # Small gap between top 2 classes = model nearly undecided
    probs = sorted(
        payload.get("probabilities", {
            "Bullish": 33.0, "Neutral": 34.0, "Bearish": 33.0
        }).values(),
        reverse=True,
    )
    margin = probs[0] - probs[1] if len(probs) >= 2 else 0.0

    if margin < 15:
        score += 0.20
        factors.append(
            f"Very thin probability margin ({margin:.1f}%) — "
            f"model nearly undecided between outcomes"
        )
    elif margin < 25:
        score += 0.10
        factors.append(f"Narrow probability margin ({margin:.1f}%)")

    # ── 4. News vs Technical conflict ─────────────────────────
    prediction = payload.get("prediction", "Neutral")
    sentiment  = payload.get("sentiment_score", 0.0)

    # Use pre-computed conflict flag from coordinator if available
    conflict = payload.get("signals_conflicting", False)
    if not conflict:
        # Recompute locally as fallback
        if prediction == "Bullish" and sentiment < -0.4:
            conflict = True
            factors.append("Negative news sentiment conflicts with Bullish prediction")
        elif prediction == "Bearish" and sentiment > 0.4:
            conflict = True
            factors.append("Positive news sentiment conflicts with Bearish prediction")

    if conflict:
        score += 0.15

    # ── 5. Index divergence ───────────────────────────────────
    nifty     = payload.get("nifty_trend",     "Neutral")
    banknifty = payload.get("banknifty_trend", "Neutral")
    sensex    = payload.get("sensex_trend",    "Neutral")
    trends    = [nifty, banknifty, sensex]

    if trends.count("Bullish") > 0 and trends.count("Bearish") > 0:
        score += 0.10
        factors.append(
            f"Mixed macro signals — "
            f"Nifty:{nifty} BankNifty:{banknifty} Sensex:{sensex}"
        )

    # ── 6. Market breadth ─────────────────────────────────────
    breadth = payload.get("market_breadth", "Mixed")
    if breadth == "Weak":
        score += 0.10
        factors.append("Weak market breadth — broad market falling")
    elif breadth == "Mixed":
        score += 0.05

    # ── 7. High impact event ──────────────────────────────────
    impact_score   = payload.get("impact_score",   0.0)
    dominant_event = payload.get("dominant_event", "Other")

    if impact_score > 0.7:
        score += 0.10
        factors.append(
            f"High impact {dominant_event} event — "
            f"outcome uncertainty elevated"
        )

    # ── 8. Abnormal volume ────────────────────────────────────
    vol_ratio = payload.get("volume_ratio", 1.0)
    if vol_ratio > 2.5:
        score += 0.10
        factors.append(
            f"Abnormal volume spike ({vol_ratio:.1f}x average) — "
            f"unusual activity detected"
        )

    # ── 9. S/R breakout zone ──────────────────────────────────
    sr_signal = payload.get("sr_signal", "None")
    if sr_signal in ["Breakout", "Breakdown"]:
        score += 0.05
        factors.append(
            f"{sr_signal} detected — price at key level, "
            f"potential reversal zone"
        )

    # ── Clamp and classify ────────────────────────────────────
    score = round(min(score, 1.0), 2)

    if score < 0.35:
        risk       = "Low"
        suggestion = "Reasonably safe to act on this prediction"
    elif score < 0.65:
        risk       = "Medium"
        suggestion = "Proceed with caution — use tight stop loss"
    else:
        risk       = "High"
        suggestion = "High uncertainty — wait for confirmation signal"

    confidence_reliable = (
        confidence >= MEDIUM_CONFIDENCE_THRESHOLD
        and margin >= 25
        and not conflict
    )

    result = {
        "risk":                risk,
        "risk_score":          score,
        "risk_factors":        factors if factors else ["No significant risk factors identified"],
        "confidence_reliable": confidence_reliable,
        "probability_margin":  round(margin, 2),
        "trade_suggestion":    suggestion,
    }

    logger.info(
        f"[Risk Agent] {company.upper()} → "
        f"Risk: {risk} ({score}) | "
        f"Margin: {margin:.1f}% | "
        f"Factors: {len(factors)}"
    )
    return result