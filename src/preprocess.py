"""
Layer 1 + 2 orchestration: data ingestion and preprocessing pipeline.
Reusable processing logic lives under utils/.
"""

import sys
from pathlib import Path

import joblib
import numpy as np

# Allow running as script: `python src/preprocess.py`
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.constants import (
    DROP_COLS,
    MODELS_DIR,
    PIPELINE_PATH,
    PROCESSED_DIR,
    RANDOM_STATE,
    RAW_DATA_PATH,
    SMOTE_K_NEIGHBORS,
    TARGET_COL,
)
from utils.features import engineer_features_df
from utils.io_utils import ensure_directories, load_csv, save_splits_as_npy
from utils.preprocessing_utils import (
    apply_smote_to_train,
    build_scaler_pipeline,
    encode_binary_target,
    get_feature_columns,
    stratified_train_val_test_split,
)


def run_preprocessing():
    ensure_directories([PROCESSED_DIR, MODELS_DIR])

    df = load_csv(RAW_DATA_PATH)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    df = encode_binary_target(df, TARGET_COL)
    print(f"Target encoded — class distribution:\n{df[TARGET_COL].value_counts().to_string()}")

    df = engineer_features_df(df)
    print(f"Feature engineering done — shape now: {df.shape}")

    feature_cols = get_feature_columns(df, DROP_COLS, TARGET_COL)
    print(f"\nFeature columns ({len(feature_cols)}):\n{feature_cols}")

    X = df[feature_cols].values
    y = df[TARGET_COL].values

    X_train, X_val, X_test, y_train, y_val, y_test = stratified_train_val_test_split(
        X=X, y=y, random_state=RANDOM_STATE
    )
    print(f"\nSplit sizes — train: {len(y_train)}, val: {len(y_val)}, test: {len(y_test)}")

    pipeline = build_scaler_pipeline()
    X_train_scaled = pipeline.fit_transform(X_train)
    X_val_scaled = pipeline.transform(X_val)
    X_test_scaled = pipeline.transform(X_test)

    X_train_balanced, y_train_balanced = apply_smote_to_train(
        X_train_scaled, y_train, random_state=RANDOM_STATE, k_neighbors=SMOTE_K_NEIGHBORS
    )
    print(f"SMOTE applied — {len(y_train)} -> {len(y_train_balanced)} training samples")
    print(f"Class distribution after SMOTE: {np.bincount(y_train_balanced)}")

    save_splits_as_npy(
        PROCESSED_DIR,
        {
            "X_train": X_train_balanced,
            "y_train": y_train_balanced,
            "X_val": X_val_scaled,
            "y_val": y_val,
            "X_test": X_test_scaled,
            "y_test": y_test,
        },
    )
    joblib.dump(feature_cols, f"{PROCESSED_DIR}/feature_cols.pkl")
    joblib.dump(pipeline, PIPELINE_PATH)

    print("\nPreprocessing complete.")
    print(f"Pipeline saved to: {PIPELINE_PATH}")
    print(f"Processed data saved to: {PROCESSED_DIR}/")
    return {
        "X_train": X_train_balanced,
        "y_train": y_train_balanced,
        "X_val": X_val_scaled,
        "y_val": y_val,
        "X_test": X_test_scaled,
        "y_test": y_test,
        "feature_cols": feature_cols,
        "pipeline": pipeline,
    }


if __name__ == "__main__":
    run_preprocessing()
