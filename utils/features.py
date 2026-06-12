"""Shared feature engineering utilities for MetroPT-3."""

from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

from utils.constants import ANALOGUE_WINDOW_FEATURES, DIGITAL_WINDOW_FEATURES


REQUIRED_SENSOR_FIELDS = [
    "TP2",
    "TP3",
    "H1",
    "DV_pressure",
    "Reservoirs",
    "Oil_temperature",
    "Motor_current",
    "COMP",
    "DV_eletric",
    "Towers",
    "MPG",
    "LPS",
    "Pressure_switch",
    "Oil_level",
    "Caudal_impulses",
]


def _validate_required_columns(df: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_SENSOR_FIELDS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for MetroPT-3 feature engineering: {missing}")


def engineer_row_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived row-level features before time-window aggregation."""
    _validate_required_columns(df)
    out = df.copy()
    out["pressure_drop"] = out["TP3"] - out["TP2"]
    out["pressure_ratio"] = out["TP2"] / (out["TP3"] + 1e-6)
    out["reservoir_vs_panel"] = out["Reservoirs"] - out["TP3"]
    out["temp_current_product"] = out["Oil_temperature"] * out["Motor_current"]
    out["temp_normalised"] = out["Oil_temperature"] / (out["Motor_current"] + 1e-6)
    out["compressor_active"] = ((out["COMP"] == 0) & (out["DV_eletric"] == 1)).astype(float)
    out["load_indicator"] = out["Motor_current"] * out["compressor_active"]
    return out


def engineer_single_record(record: Mapping[str, Any]) -> dict[str, float]:
    """Engineer derived row-level features for a single inference record."""
    row = {key: float(record[key]) for key in REQUIRED_SENSOR_FIELDS}
    df = engineer_row_features(pd.DataFrame([row]))
    return {key: float(value) for key, value in df.iloc[0].to_dict().items()}


def approximate_window_features(record: Mapping[str, Any]) -> dict[str, float]:
    """
    Approximate rolling-window features from one sensor reading.

    Used by API inference when no temporal buffer is provided.
    """
    engineered = engineer_single_record(record)
    window_features: dict[str, float] = {}

    for col in ANALOGUE_WINDOW_FEATURES:
        value = float(engineered[col])
        window_features[f"{col}_mean"] = value
        window_features[f"{col}_std"] = 0.0
        window_features[f"{col}_min"] = value
        window_features[f"{col}_max"] = value
        window_features[f"{col}_range"] = 0.0

    for col in DIGITAL_WINDOW_FEATURES:
        window_features[f"{col}_prop"] = float(engineered[col])

    return window_features

