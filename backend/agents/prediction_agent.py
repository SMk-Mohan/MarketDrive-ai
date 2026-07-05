"""
agents/prediction_agent.py
Trains XGBoost models (one per stock) and runs live inference
with SHAP explanations and TimeSeriesSplit validation.
"""

import os
import json
import logging
import asyncio
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import shap
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
from functools import partial

from config.settings import (
    MODEL_DIR,
    ENCODER_DIR,
    TRAINING_DATA_DIR,
    FEATURE_COLUMNS,
    CATEGORICAL_COLUMNS,
    MIN_TRAINING_ROWS,
    RANDOM_STATE,
)

logger = logging.getLogger(__name__)

os.makedirs(MODEL_DIR,   exist_ok=True)
os.makedirs(ENCODER_DIR, exist_ok=True)

TARGET_COLUMN = "target"   # -1 Bearish, 0 Neutral, 1 Bullish

# XGBoost label mapping: model sees 0,1,2 internally
LABEL_MAP         = {-1: 0,  0: 1,  1: 2}
INVERSE_LABEL_MAP = { 0: -1, 1: 0,  2: 1}
LABEL_NAMES       = { 0: "Bearish", 1: "Neutral", 2: "Bullish" }

XGB_PARAMS = dict(
    n_estimators=150, max_depth=4,
    learning_rate=0.05, subsample=0.8,
    colsample_bytree=0.8, reg_alpha=0.5,
    reg_lambda=1.0, objective="multi:softprob",
    num_class=3, eval_metric="mlogloss",
    random_state=RANDOM_STATE,
)


# ── Data Loader ───────────────────────────────────────────────

def load_training_csv(csv_path: str) -> pd.DataFrame:
    """
    Loads and sorts training CSV chronologically.
    Time order is critical for TimeSeriesSplit.
    """
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    logger.info(
        f"[Prediction Agent] Loaded {csv_path} → {len(df)} rows "
        f"({df['date'].min().date()} → {df['date'].max().date()})"
    )
    if len(df) < MIN_TRAINING_ROWS:
        raise ValueError(
            f"Insufficient training data: {len(df)} rows "
            f"(minimum {MIN_TRAINING_ROWS})"
        )
    return df


# ── Categorical Encoder ───────────────────────────────────────

def encode_categoricals(df: pd.DataFrame, company_name: str,
                         fit: bool = True) -> pd.DataFrame:
    """
    Encodes categorical columns to *_enc integer columns.
    fit=True  → training: fits + saves encoder pkl files
    fit=False → inference: loads saved encoder pkl files

    Unseen categories are mapped to "Unknown" — never crashes.
    """
    df = df.copy()
    for col in CATEGORICAL_COLUMNS:
        enc_path = os.path.join(ENCODER_DIR,
                                f"{company_name}_{col}_encoder.pkl")
        if fit:
            le = LabelEncoder()
            vals = df[col].astype(str).tolist() + ["Unknown"]
            le.fit(vals)
            joblib.dump(le, enc_path)
        else:
            if not os.path.exists(enc_path):
                raise FileNotFoundError(
                    f"Encoder not found: {enc_path} — "
                    f"run training first."
                )
            le = joblib.load(enc_path)

        df[col] = df[col].astype(str).apply(
            lambda x: x if x in le.classes_ else "Unknown"
        )
        df[f"{col}_enc"] = le.transform(df[col])
    return df


from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import classification_report, f1_score

# ── Model Trainer ─────────────────────────────────────────────

def train_model(company_name: str, csv_path: str) -> dict:
    """
    Trains XGBoost for one stock with imbalance handling and CV.
    """
    logger.info(f" TRAINING — {company_name.upper()}")

    df = load_training_csv(csv_path)
    df = encode_categoricals(df, company_name, fit=True)
    X  = df[FEATURE_COLUMNS]
    y  = df[TARGET_COLUMN].map(LABEL_MAP)

    logger.info(f"Class distribution:\n"
          f"{y.value_counts(normalize=True).rename({0:'Bearish',1:'Neutral',2:'Bullish'}).to_string()}")

    # ── Walk-forward validation ───────────────────────────────
    tscv = TimeSeriesSplit(n_splits=5)
    fold_f1s = []
    
    # Track overall true/pred for a final report
    all_y_true = []
    all_y_pred = []

    # Walk-forward validation
    for fold, (tr_idx, te_idx) in enumerate(tscv.split(X), 1):
        X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
        X_te, y_te = X.iloc[te_idx], y.iloc[te_idx]
        
        # Calculate weights for this fold's training set
        weights = compute_sample_weight(class_weight='balanced', y=y_tr)
        
        m = xgb.XGBClassifier(**XGB_PARAMS)
        m.fit(X_tr, y_tr, sample_weight=weights)
        
        preds = m.predict(X_te)
        f1 = f1_score(y_te, preds, average='macro', zero_division=0)
        fold_f1s.append(f1)
        
        all_y_true.extend(y_te.tolist())
        all_y_pred.extend(preds.tolist())
        
        # Training fold

    # Final Model Report
    report = classification_report(
        all_y_true, all_y_pred, 
        target_names=['Bearish', 'Neutral', 'Bullish'],
        zero_division=0
    )
    # Cleanup

    # ── Final model on all data ───────────────────────────────
    final_weights = compute_sample_weight(class_weight='balanced', y=y)
    final_model = xgb.XGBClassifier(**XGB_PARAMS)
    final_model.fit(X, y, sample_weight=final_weights)

    model_path = os.path.join(MODEL_DIR, f"{company_name}_model.pkl")
    meta_path  = os.path.join(MODEL_DIR, f"{company_name}_metadata.json")

    joblib.dump(final_model, model_path)

    metadata = {
        "company":               company_name,
        "trained_at":            datetime.now().isoformat(),
        "rows_trained_on":       len(df),
        "walk_forward_f1":       round(float(np.mean(fold_f1s)), 4),
        "feature_columns":       FEATURE_COLUMNS,
    }
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return metadata


# ── Live Prediction ───────────────────────────────────────────

def predict(company_name: str, live_features: dict) -> dict:
    """
    Runs live prediction for one stock.

    Args:
        company_name : e.g. "infosys"
        live_features: flat dict with raw feature values
                       Keys must include all FEATURE_COLUMNS
                       plus raw CATEGORICAL_COLUMNS before encoding.

    Returns:
        Full prediction dict with direction, confidence,
        probabilities, price range, SHAP top features, reason.
    """
    model_path = os.path.join(MODEL_DIR, f"{company_name}_model.pkl")
    meta_path  = os.path.join(MODEL_DIR, f"{company_name}_metadata.json")

    if not os.path.exists(model_path):
        logger.error(f"[Prediction Agent] No model for {company_name}")
        return {
            "error":       f"No trained model for {company_name}. Run training first.",
            "prediction":  "Neutral",
            "confidence":  33.0,
            "company":     company_name,
            "timestamp":   datetime.now().isoformat(),
        }

    model = joblib.load(model_path)
    with open(meta_path) as f:
        metadata = json.load(f)

    # Build single-row DataFrame and encode
    df_live = pd.DataFrame([live_features.copy()])
    df_live = encode_categoricals(df_live, company_name, fit=False)
    X_live  = df_live[FEATURE_COLUMNS]

    # ── Inference ─────────────────────────────────────────────
    probs      = model.predict_proba(X_live)[0]
    pred_class = int(np.argmax(probs))
    prediction = LABEL_NAMES[pred_class]
    confidence = round(float(probs[pred_class]) * 100, 2)

    # ── Price range via ATR ───────────────────────────────────
    price      = float(live_features.get("close", 0.0))
    atr        = float(live_features.get("atr",   0.0))
    price_low  = round(price - atr * 0.5, 2)
    price_high = round(price + atr * 0.5, 2)

    # ── SHAP explanation ──────────────────────────────────────
    top_features = []
    reason       = f"Predicted {prediction} with {confidence:.1f}% confidence"

    try:
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_live)

        # FIX: handle both old (list) and new (3D ndarray) xgboost SHAP output
        if isinstance(shap_values, list):
            # older xgboost: list of arrays, one per class
            class_shap = np.array(shap_values[pred_class][0])
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            # newer xgboost: shape (n_samples, n_features, n_classes)
            class_shap = shap_values[0, :, pred_class]
        else:
            # fallback: flat array
            class_shap = np.array(shap_values[0])

        shap_pairs = sorted(
            zip(FEATURE_COLUMNS, class_shap),
            key=lambda x: abs(x[1]),
            reverse=True,
        )
        top_features = [
            {
                "feature":    feat,
                "shap_value": round(float(val), 4),
                "direction":  "supporting" if val > 0 else "opposing",
            }
            for feat, val in shap_pairs[:5]
        ]
        reason = (
            f"{prediction} driven by: "
            + ", ".join(
                f"{f['feature']} ({f['direction']}, {f['shap_value']:+.3f})"
                for f in top_features
            )
        )
    except Exception as e:
        logger.warning(f"[Prediction Agent] SHAP error: {e}")

    return {
        "company":          company_name,
        "prediction":       prediction,
        "confidence":       confidence,
        "probabilities": {
            "Bearish": round(float(probs[0]) * 100, 2),
            "Neutral": round(float(probs[1]) * 100, 2),
            "Bullish": round(float(probs[2]) * 100, 2),
        },
        "current_price":    price,
        "price_range_low":  price_low,
        "price_range_high": price_high,
        "reason":           reason,
        "top_features":     top_features,
        "model_accuracy":   metadata["walk_forward_accuracy"],
        "model_trained_at": metadata["trained_at"],
        "timestamp":        datetime.now().isoformat(),
    }


# ── Incremental retraining ────────────────────────────────────

def retrain_incremental(company_name: str, csv_path: str) -> dict:
    """
    Retrains on updated CSV (new rows appended since last run).
    Called by scheduler at RETRAIN_TIME (4:30 PM IST).
    """
    logger.info(f"[Prediction Agent] Incremental retrain: {company_name}")
    return train_model(company_name, csv_path)


# ── Async wrapper for coordinator ─────────────────────────────

async def run_prediction(company_name: str,
                          live_features: dict) -> dict:
    """
    Async wrapper — runs sync predict() in thread pool.
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        partial(predict, company_name, live_features)
    )