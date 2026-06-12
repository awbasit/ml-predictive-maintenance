"""Reusable preprocessing helpers for MetroPT-3 pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from utils.constants import (
    ANALOGUE_WINDOW_FEATURES,
    DIGITAL_WINDOW_FEATURES,
    FAULT_WINDOWS,
    HORIZON_OPTIONS,
    RESAMPLE_RULE,
    ROLLING_WINDOW_BINS,
)


def load_metropt_data(path: str) -> pd.DataFrame:
    """Load MetroPT-3 dataset with timestamp parsing and cleanup."""
    df = pd.read_csv(path, parse_dates=["timestamp"], low_memory=False)
    df = df.drop(columns=["Unnamed: 0"], errors="ignore")
    df = df.sort_values("timestamp").dropna().reset_index(drop=True)
    return df


def engineer_temporal_labels(df: pd.DataFrame, horizon_key: str) -> pd.DataFrame:
    """Create binary labels from documented fault windows and pre-fault horizon."""
    if horizon_key not in HORIZON_OPTIONS:
        raise ValueError(f"Unsupported horizon '{horizon_key}'. Choose from {list(HORIZON_OPTIONS)}")

    out = df.copy()
    out["label"] = 0
    horizon = pd.Timedelta(hours=HORIZON_OPTIONS[horizon_key])
    for start_str, end_str in FAULT_WINDOWS:
        start = pd.Timestamp(start_str)
        end = pd.Timestamp(end_str)
        mask = (out["timestamp"] >= start - horizon) & (out["timestamp"] <= end)
        out.loc[mask, "label"] = 1
    return out


def create_window_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create rolling-window feature vectors from row-level engineered data."""
    indexed = df.set_index("timestamp")
    feature_cols = ANALOGUE_WINDOW_FEATURES + DIGITAL_WINDOW_FEATURES
    resampled = indexed[feature_cols + ["label"]].resample(RESAMPLE_RULE).agg(
        {**{col: "mean" for col in feature_cols}, "label": "max"}
    )
    resampled = resampled.dropna()

    result_frames = []
    for col in ANALOGUE_WINDOW_FEATURES:
        roll = resampled[col].rolling(window=ROLLING_WINDOW_BINS, min_periods=ROLLING_WINDOW_BINS)
        result_frames.append(roll.mean().rename(f"{col}_mean"))
        result_frames.append(roll.std().rename(f"{col}_std"))
        result_frames.append(roll.min().rename(f"{col}_min"))
        result_frames.append(roll.max().rename(f"{col}_max"))
        result_frames.append((roll.max() - roll.min()).rename(f"{col}_range"))

    for col in DIGITAL_WINDOW_FEATURES:
        roll = resampled[col].rolling(window=ROLLING_WINDOW_BINS, min_periods=ROLLING_WINDOW_BINS)
        result_frames.append(roll.mean().rename(f"{col}_prop"))

    label_roll = resampled["label"].rolling(window=ROLLING_WINDOW_BINS, min_periods=ROLLING_WINDOW_BINS).max()
    result_frames.append(label_roll.rename("label"))
    out = pd.concat(result_frames, axis=1).dropna().reset_index(drop=True)
    out["label"] = out["label"].astype(int)
    return out


def build_scaler_pipeline() -> Pipeline:
    """Return standard preprocessing pipeline for numeric features."""
    return Pipeline([("scaler", StandardScaler())])


def stratified_train_val_test_split(
    X: np.ndarray,
    y: np.ndarray,
    random_state: int,
    train_size: float = 0.70,
    val_size: float = 0.15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split into train/val/test with stratification.

    Defaults implement 70/15/15.
    """
    test_plus_val = 1.0 - train_size
    val_within_temp = val_size / test_plus_val
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=test_plus_val, random_state=random_state, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, train_size=val_within_temp, random_state=random_state, stratify=y_temp
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def apply_smote_to_train(
    X_train: np.ndarray, y_train: np.ndarray, random_state: int, k_neighbors: int
) -> tuple[np.ndarray, np.ndarray]:
    """Apply SMOTE only to training split."""
    smote = SMOTE(random_state=random_state, k_neighbors=k_neighbors)
    return smote.fit_resample(X_train, y_train)

