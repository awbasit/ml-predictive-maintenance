"""
Layer 4: FastAPI inference service.
Exposes POST /predict and GET /health endpoints.
Loads the best model and preprocessing pipeline at startup.
"""

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal
import os

from utils.features import engineer_features_record
from utils.io_utils import load_inference_artifacts

app = FastAPI(
    title="AC Compressor Predictive Maintenance API",
    description="Predicts bearing health status from compressor sensor readings.",
    version="1.0.0",
)

# Globals — loaded once at startup
pipeline = None
model = None
feature_cols = None
model_meta = None


@app.on_event("startup")
def load_artifacts():
    global pipeline, model, feature_cols, model_meta

    models_dir = os.getenv("MODELS_DIR", "models")
    processed_dir = os.getenv("PROCESSED_DIR", "data/processed")

    try:
        pipeline, model, feature_cols, model_meta = load_inference_artifacts(models_dir, processed_dir)
        print(f"Loaded model: {model_meta['name']} | Val F1: {model_meta['val_f1']}")
    except Exception as e:
        print(f"WARNING: Could not load artifacts — {e}")


class SensorReading(BaseModel):
    rpm: float = Field(..., description="Motor RPM", example=1499.0)
    motor_power: float = Field(..., description="Motor power (kW)", example=6984.0)
    torque: float = Field(..., description="Torque (Nm)", example=49.19)
    outlet_pressure_bar: float = Field(..., description="Outlet pressure (bar)", example=4.05)
    air_flow: float = Field(..., description="Air flow (m3/min)", example=754.67)
    noise_db: float = Field(..., description="Noise level (dB)", example=53.41)
    outlet_temp: float = Field(..., description="Outlet temperature (C)", example=118.86)
    wpump_outlet_press: float = Field(..., description="Water pump outlet pressure (bar)", example=2.80)
    water_inlet_temp: float = Field(..., description="Water inlet temperature (C)", example=83.02)
    water_outlet_temp: float = Field(..., description="Water outlet temperature (C)", example=96.64)
    wpump_power: float = Field(..., description="Water pump power (kW)", example=222.19)
    water_flow: float = Field(..., description="Water flow (m3/min)", example=53.71)
    oilpump_power: float = Field(..., description="Oil pump power (kW)", example=300.48)
    oil_tank_temp: float = Field(..., description="Oil tank temperature (C)", example=46.24)
    gaccx: float = Field(..., description="Ground acceleration X", example=0.60)
    gaccy: float = Field(..., description="Ground acceleration Y", example=0.35)
    gaccz: float = Field(..., description="Ground acceleration Z", example=3.92)
    haccx: float = Field(..., description="Head acceleration X", example=1.10)
    haccy: float = Field(..., description="Head acceleration Y", example=1.35)
    haccz: float = Field(..., description="Head acceleration Z", example=3.50)


class PredictionResponse(BaseModel):
    status: Literal["Ok", "Noisy"]
    probability_noisy: float
    probability_ok: float
    alert: bool
    alert_message: str
    model_used: str


def engineer_single(reading: SensorReading) -> dict:
    """Replicate feature engineering for a single inference record."""
    return engineer_features_record(reading.model_dump())


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
        engineered = engineer_single(reading)
        X_raw = np.array([[engineered[col] for col in feature_cols]])
        X_scaled = pipeline.transform(X_raw)
        proba = model.predict_proba(X_scaled)[0]
        prob_noisy = float(proba[1])
        prob_ok = float(proba[0])
        prediction = int(model.predict(X_scaled)[0])
        status = "Noisy" if prediction == 1 else "Ok"
        alert = prob_noisy >= 0.5

        alert_message = (
            f"WARNING: Bearing degradation detected (confidence: {prob_noisy:.1%}). "
            "Schedule maintenance inspection."
            if alert else
            f"Bearings operating normally (confidence: {prob_ok:.1%})."
        )

        return PredictionResponse(
            status=status,
            probability_noisy=round(prob_noisy, 4),
            probability_ok=round(prob_ok, 4),
            alert=alert,
            alert_message=alert_message,
            model_used=model_meta["name"] if model_meta else "unknown",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def root():
    return {
        "service": "AC Compressor Predictive Maintenance API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }
