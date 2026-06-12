"""Reusable preprocessing helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def encode_binary_target(df: pd.DataFrame, target_col: str, positive_label: str = "Noisy") -> pd.DataFrame:
    """Encode target as binary with positive_label mapped to 1."""
    out = df.copy()
    out[target_col] = (out[target_col] == positive_label).astype(int)
    return out


def get_feature_columns(df: pd.DataFrame, drop_cols: list[str], target_col: str) -> list[str]:
    """Return model feature columns by excluding target and dropped columns."""
    exclude = set(drop_cols + [target_col])
    return [col for col in df.columns if col not in exclude]


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

