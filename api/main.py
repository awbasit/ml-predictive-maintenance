"""
Layer 8: FastAPI inference service for MetroPT-3 predictive maintenance.
"""

from __future__ import annotations

import json
import os
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Literal

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from utils.features import approximate_window_features
from utils.io_utils import load_inference_artifacts

# ─── Global state ─────────────────────────────────────────────────────────────
pipeline = None
model = None
feature_cols = None
model_meta = None
eval_results = None
feature_importances: list[tuple[str, float]] | None = None
prediction_log: deque[dict] = deque(maxlen=100)


def _load_all_artifacts() -> None:
    global pipeline, model, feature_cols, model_meta, eval_results, feature_importances
    models_dir = os.getenv("MODELS_DIR", "models")
    processed_dir = os.getenv("PROCESSED_DIR", "data/processed")

    try:
        pipeline, model, feature_cols, model_meta = load_inference_artifacts(
            models_dir, processed_dir
        )
        print(f"Loaded model: {model_meta['name']} | Val F1: {model_meta['val_f1']}")
    except Exception as exc:
        print(f"WARNING: Could not load model artifacts — {exc}")

    eval_path = os.path.join(models_dir, "evaluation_results.json")
    if os.path.exists(eval_path):
        with open(eval_path, encoding="utf-8") as fh:
            eval_results = json.load(fh)
        print("Loaded evaluation results.")
    else:
        print(f"WARNING: {eval_path} not found.")

    if model is not None and hasattr(model, "feature_importances_") and feature_cols:
        pairs = sorted(
            zip(feature_cols, model.feature_importances_.tolist()),
            key=lambda x: x[1],
            reverse=True,
        )
        feature_importances = [(name, float(imp)) for name, imp in pairs[:20]]
        print(f"Extracted {len(feature_importances)} feature importances.")


# ─── Lifespan ─────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_all_artifacts()
    yield


# ─── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Compressor Predictive Maintenance API",
    description=(
        "Real-time fault prediction from MetroPT-3 sensor readings. "
        "Trained on 1.5M rows of industrial air compressor data, "
        "windowed into ~100K feature vectors using a 10-minute rolling strategy."
    ),
    version="3.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Schemas ──────────────────────────────────────────────────────────────────
class SensorReading(BaseModel):
    TP2: float = Field(..., description="Compressor pressure (bar)", example=-0.012)
    TP3: float = Field(..., description="Pneumatic panel pressure (bar)", example=9.358)
    H1: float = Field(..., description="Cyclonic separator pressure (bar)", example=9.340)
    DV_pressure: float = Field(..., description="Air dryer pressure drop (bar)", example=-0.024)
    Reservoirs: float = Field(..., description="Downstream reservoir pressure (bar)", example=9.358)
    Oil_temperature: float = Field(..., description="Compressor oil temperature (°C)", example=53.6)
    Motor_current: float = Field(..., description="Motor phase current (A)", example=0.04)
    COMP: float = Field(..., description="Air intake valve signal (0/1)", example=1.0)
    DV_eletric: float = Field(..., description="Outlet valve signal (0/1)", example=0.0)
    Towers: float = Field(..., description="Drying tower selector (0/1)", example=1.0)
    MPG: float = Field(..., description="Load start signal (0/1)", example=1.0)
    LPS: float = Field(..., description="Low pressure switch (0/1)", example=0.0)
    Pressure_switch: float = Field(..., description="Pressure switch signal (0/1)", example=1.0)
    Oil_level: float = Field(..., description="Oil level signal (0/1)", example=1.0)
    Caudal_impulses: float = Field(..., description="Air flow pulse signal (0/1)", example=1.0)


class PredictionResponse(BaseModel):
    status: Literal["Normal", "Fault"]
    probability_fault: float
    probability_normal: float
    alert: bool
    alert_message: str
    model_used: str
    horizon: str
    timestamp: str
    top_features: list[dict]


class HealthResponse(BaseModel):
    status: Literal["healthy", "degraded"]
    model_loaded: bool
    model_name: str | None
    model_val_f1: float | None
    predictions_served: int


# ─── Routes ───────────────────────────────────────────────────────────────────
@app.get("/", tags=["Info"])
def root():
    return {
        "service": "Compressor Predictive Maintenance API",
        "version": "3.0.0",
        "docs": "/docs",
        "endpoints": {
            "health": "GET /health",
            "predict": "POST /predict",
            "metrics": "GET /metrics",
            "feature_importance": "GET /feature-importance",
            "history": "GET /history",
        },
    }


@app.get("/health", response_model=HealthResponse, tags=["Info"])
def health():
    loaded = model is not None and pipeline is not None
    return HealthResponse(
        status="healthy" if loaded else "degraded",
        model_loaded=loaded,
        model_name=model_meta["name"] if model_meta else None,
        model_val_f1=model_meta["val_f1"] if model_meta else None,
        predictions_served=len(prediction_log),
    )


@app.get("/metrics", tags=["Evaluation"])
def get_metrics():
    """Return full evaluation results for all trained models."""
    if eval_results is None:
        raise HTTPException(status_code=404, detail="Evaluation results not available.")
    return eval_results


@app.get("/feature-importance", tags=["Evaluation"])
def get_feature_importance():
    """Return top feature importances for the loaded model (tree models only)."""
    if feature_importances is None:
        raise HTTPException(
            status_code=404,
            detail="Feature importances not available — model may not support them.",
        )
    return {
        "model": model_meta["name"] if model_meta else "unknown",
        "importances": feature_importances,
    }


@app.get("/history", tags=["Predictions"])
def get_history():
    """Return the last 100 predictions served in this session."""
    return {"count": len(prediction_log), "predictions": list(prediction_log)}


@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
def predict(reading: SensorReading):
    """
    Predict compressor fault probability from a single sensor snapshot.

    Window features are approximated by treating the single reading as
    representing the entire 10-minute window (std=0, range=0).
    """
    if model is None or pipeline is None:
        raise HTTPException(status_code=503, detail="Model artifacts not loaded.")

    try:
        window_features = approximate_window_features(reading.model_dump())
        X_raw = np.array([[window_features.get(col, 0.0) for col in feature_cols]])
        X_scaled = pipeline.transform(X_raw)

        proba = model.predict_proba(X_scaled)[0]
        prob_fault = float(proba[1])
        prob_normal = float(proba[0])
        pred = int(model.predict(X_scaled)[0])

        status: Literal["Normal", "Fault"] = "Fault" if pred == 1 else "Normal"
        alert = prob_fault >= 0.5
        alert_message = (
            f"ALERT: Compressor fault likely (confidence: {prob_fault:.1%})."
            if alert
            else f"System operating normally (confidence: {prob_normal:.1%})."
        )

        now = datetime.now(timezone.utc).isoformat()

        top_features = (
            [{"feature": name, "importance": round(imp, 5)} for name, imp in feature_importances[:5]]
            if feature_importances
            else []
        )

        response = PredictionResponse(
            status=status,
            probability_fault=round(prob_fault, 4),
            probability_normal=round(prob_normal, 4),
            alert=alert,
            alert_message=alert_message,
            model_used=model_meta["name"] if model_meta else "unknown",
            horizon="6hr",
            timestamp=now,
            top_features=top_features,
        )

        prediction_log.append(
            {
                "timestamp": now,
                "status": status,
                "probability_fault": round(prob_fault, 4),
                "alert": alert,
            }
        )

        return response

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
