"""
Layer 8: FastAPI inference service for MetroPT-3 predictive maintenance.
"""

from __future__ import annotations

import os
from typing import Literal

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from utils.features import approximate_window_features
from utils.io_utils import load_inference_artifacts

app = FastAPI(
    title="Compressor Predictive Maintenance API",
    description="Predict compressor fault probability from MetroPT-3 sensor readings.",
    version="2.0.0",
)

pipeline = None
model = None
feature_cols = None
model_meta = None


@app.on_event("startup")
def load_artifacts():
    """Load serialized artifacts once when service starts."""
    global pipeline, model, feature_cols, model_meta
    models_dir = os.getenv("MODELS_DIR", "models")
    processed_dir = os.getenv("PROCESSED_DIR", "data/processed")
    try:
        pipeline, model, feature_cols, model_meta = load_inference_artifacts(models_dir, processed_dir)
        print(f"Loaded model: {model_meta['name']} | Val F1: {model_meta['val_f1']}")
    except Exception as exc:
        print(f"WARNING: Could not load artifacts — {exc}")


class SensorReading(BaseModel):
    TP2: float = Field(..., description="Compressor pressure", example=-0.012)
    TP3: float = Field(..., description="Pneumatic panel pressure", example=9.358)
    H1: float = Field(..., description="Cyclonic separator pressure", example=9.340)
    DV_pressure: float = Field(..., description="Air dryer pressure drop", example=-0.024)
    Reservoirs: float = Field(..., description="Downstream reservoir pressure", example=9.358)
    Oil_temperature: float = Field(..., description="Compressor oil temperature", example=53.6)
    Motor_current: float = Field(..., description="Motor phase current", example=0.04)
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


@app.get("/")
def root():
    return {
        "service": "Compressor Predictive Maintenance API",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health():
    loaded = model is not None and pipeline is not None
    return {
        "status": "healthy" if loaded else "degraded",
        "model_loaded": loaded,
        "model_name": model_meta["name"] if model_meta else None,
        "model_val_f1": model_meta["val_f1"] if model_meta else None,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(reading: SensorReading):
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
        status = "Fault" if pred == 1 else "Normal"
        alert = prob_fault >= 0.5
        alert_message = (
            f"ALERT: Compressor fault likely (confidence: {prob_fault:.1%})."
            if alert
            else f"System operating normally (confidence: {prob_normal:.1%})."
        )

        return PredictionResponse(
            status=status,
            probability_fault=round(prob_fault, 4),
            probability_normal=round(prob_normal, 4),
            alert=alert,
            alert_message=alert_message,
            model_used=model_meta["name"] if model_meta else "unknown",
            horizon="6hr",
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
