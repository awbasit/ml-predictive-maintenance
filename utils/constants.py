"""Project-wide constants used across modules."""

RAW_DATA_PATH = "data/raw_data.csv"
PROCESSED_DIR = "data/processed"
MODELS_DIR = "models"
PIPELINE_PATH = f"{MODELS_DIR}/preprocessing_pipeline.pkl"

TARGET_COL = "bearings"
DROP_COLS = ["id", "wpump", "radiator", "exvalve", "acmotor"]

RANDOM_STATE = 42
SMOTE_K_NEIGHBORS = 5

