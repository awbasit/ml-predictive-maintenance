"""
Shared feature engineering utilities.

Single source of truth for derived features in training and inference.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

TEMPERATURE_FEATURES = [
    "outlet_temp",
    "water_inlet_temp",
    "water_outlet_temp",
    "oil_tank_temp",
]

BASE_SENSOR_FIELDS = [
    "rpm",
    "motor_power",
    "torque",
    "outlet_pressure_bar",
    "air_flow",
    "noise_db",
    "outlet_temp",
    "wpump_outlet_press",
    "water_inlet_temp",
    "water_outlet_temp",
    "wpump_power",
    "water_flow",
    "oilpump_power",
    "oil_tank_temp",
    "gaccx",
    "gaccy",
    "gaccz",
    "haccx",
    "haccy",
    "haccz",
]

ENGINEERED_FEATURES = [
    "temp_water_delta",
    "temp_outlet_vs_oil",
    "temp_mean",
    "temp_max",
    "power_per_rpm",
    "torque_per_rpm",
    "pressure_air_ratio",
    "vib_ground_magnitude",
    "vib_head_magnitude",
    "vib_ratio",
]


def _validate_required_columns(df: pd.DataFrame) -> None:
    missing = [col for col in BASE_SENSOR_FIELDS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required sensor columns for feature engineering: {missing}")


def engineer_features_df(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features to a dataframe with raw sensor fields."""
    _validate_required_columns(df)
    out = df.copy()

    out["temp_water_delta"] = out["water_outlet_temp"] - out["water_inlet_temp"]
    out["temp_outlet_vs_oil"] = out["outlet_temp"] - out["oil_tank_temp"]
    out["temp_mean"] = out[TEMPERATURE_FEATURES].mean(axis=1)
    out["temp_max"] = out[TEMPERATURE_FEATURES].max(axis=1)

    out["power_per_rpm"] = out["motor_power"] / (out["rpm"] + 1.0)
    out["torque_per_rpm"] = out["torque"] / (out["rpm"] + 1.0)
    out["pressure_air_ratio"] = out["outlet_pressure_bar"] / (out["air_flow"] + 1.0)

    out["vib_ground_magnitude"] = np.sqrt(out["gaccx"] ** 2 + out["gaccy"] ** 2 + out["gaccz"] ** 2)
    out["vib_head_magnitude"] = np.sqrt(out["haccx"] ** 2 + out["haccy"] ** 2 + out["haccz"] ** 2)
    out["vib_ratio"] = out["vib_head_magnitude"] / (out["vib_ground_magnitude"] + 1e-6)
    return out


def engineer_features_record(record: Mapping[str, Any]) -> dict[str, float]:
    """Engineer features for one inference record."""
    row = {key: float(record[key]) for key in BASE_SENSOR_FIELDS}
    engineered = engineer_features_df(pd.DataFrame([row])).iloc[0].to_dict()
    return {k: float(v) for k, v in engineered.items()}

