"""
Layer 1 + 2 orchestration: MetroPT-3 ingestion and preprocessing pipeline.

Label engineering:
- 3 selectable horizons: 1hr, 6hr, 24hr before each documented fault start
- Rows in fault windows + pre-fault horizon are labeled as fault (1)

Windowing:
- 10-minute rolling windows with 1-minute step
- Aggregate stats capture temporal dynamics without sequence models

Class imbalance:
- No SMOTE; downstream models use class_weight="balanced"
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np

# Allow running as script: `python src/preprocess.py`
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.constants import (
    DEFAULT_HORIZON,
    MODELS_DIR,
    PIPELINE_PATH,
    PROCESSED_DIR,
    RANDOM_STATE,
    RAW_DATA_PATH,
    TARGET_COL,
)
from utils.features import engineer_row_features
from utils.io_utils import ensure_directories, save_splits_as_npy
from utils.preprocessing_utils import (
    build_scaler_pipeline,
    create_window_features,
    engineer_temporal_labels,
    load_metropt_data,
    stratified_train_val_test_split,
)


def run_preprocessing(horizon_key: str = DEFAULT_HORIZON) -> dict:
    """Run full MetroPT-3 preprocessing pipeline and persist artifacts."""
    ensure_directories([PROCESSED_DIR, MODELS_DIR])

    df = load_metropt_data(RAW_DATA_PATH)
    print(f"Loaded {len(df):,} rows | {df['timestamp'].min()} -> {df['timestamp'].max()}")

    df = engineer_temporal_labels(df, horizon_key=horizon_key)
    fault_ratio = (df[TARGET_COL] == 1).mean() * 100
    print(f"Label engineering [{horizon_key}] fault ratio: {fault_ratio:.2f}%")

    df = engineer_row_features(df)
    windowed = create_window_features(df)
    feature_cols = [col for col in windowed.columns if col != TARGET_COL]

    X = windowed[feature_cols].values
    y = windowed[TARGET_COL].values
    X_train, X_val, X_test, y_train, y_val, y_test = stratified_train_val_test_split(
        X=X, y=y, random_state=RANDOM_STATE
    )
    print(f"Split sizes — train: {len(y_train):,}, val: {len(y_val):,}, test: {len(y_test):,}")

    pipeline = build_scaler_pipeline()
    X_train_scaled = pipeline.fit_transform(X_train)
    X_val_scaled = pipeline.transform(X_val)
    X_test_scaled = pipeline.transform(X_test)

    X_train_final, y_train_final = X_train_scaled, y_train
    print("No SMOTE applied — using class_weight='balanced' in all models.")

    save_splits_as_npy(
        PROCESSED_DIR,
        {
            "X_train": X_train_final,
            "y_train": y_train_final,
            "X_val": X_val_scaled,
            "y_val": y_val,
            "X_test": X_test_scaled,
            "y_test": y_test,
        },
    )
    joblib.dump(feature_cols, f"{PROCESSED_DIR}/feature_cols.pkl")
    joblib.dump(pipeline, PIPELINE_PATH)
    print(f"Saved preprocessing pipeline to {PIPELINE_PATH}")

    return {
        "X_train": X_train_final,
        "y_train": y_train_final,
        "X_val": X_val_scaled,
        "y_val": y_val,
        "X_test": X_test_scaled,
        "y_test": y_test,
        "feature_cols": feature_cols,
        "pipeline": pipeline,
    }


if __name__ == "__main__":
    run_preprocessing(horizon_key="6hr")
