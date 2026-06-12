"""Reusable file and artifact IO helpers."""

from __future__ import annotations

import json
import os
from typing import Any

import joblib
import numpy as np
import pandas as pd


def ensure_directories(paths: list[str]) -> None:
    """Create directories if they do not exist."""
    for path in paths:
        os.makedirs(path, exist_ok=True)


def load_csv(path: str) -> pd.DataFrame:
    """Load a CSV file."""
    return pd.read_csv(path)


def save_splits_as_npy(processed_dir: str, split_data: dict[str, np.ndarray]) -> None:
    """Persist all dataset splits to .npy files."""
    for name, values in split_data.items():
        np.save(f"{processed_dir}/{name}.npy", values)


def load_splits_from_npy(processed_dir: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load train/val/test splits from .npy files."""
    X_train = np.load(f"{processed_dir}/X_train.npy")
    y_train = np.load(f"{processed_dir}/y_train.npy")
    X_val = np.load(f"{processed_dir}/X_val.npy")
    y_val = np.load(f"{processed_dir}/y_val.npy")
    X_test = np.load(f"{processed_dir}/X_test.npy")
    y_test = np.load(f"{processed_dir}/y_test.npy")
    return X_train, y_train, X_val, y_val, X_test, y_test


def save_json(path: str, payload: Any) -> None:
    """Save JSON payload with stable formatting."""
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def load_inference_artifacts(models_dir: str, processed_dir: str):
    """Load model, preprocessing pipeline, feature list, and model metadata."""
    pipeline = joblib.load(f"{models_dir}/preprocessing_pipeline.pkl")
    model = joblib.load(f"{models_dir}/best_model.pkl")
    feature_cols = joblib.load(f"{processed_dir}/feature_cols.pkl")
    model_meta = joblib.load(f"{models_dir}/best_model_meta.pkl")
    return pipeline, model, feature_cols, model_meta

