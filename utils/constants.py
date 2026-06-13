"""Project-wide constants for MetroPT-3 pipeline."""

RAW_DATA_PATH = "data/MetroPT3_Dataset.csv"
PROCESSED_DIR = "data/processed"
MODELS_DIR = "models"
PIPELINE_PATH = f"{MODELS_DIR}/preprocessing_pipeline.pkl"
TARGET_COL = "label"

RANDOM_STATE = 42
NOISE_STD = 0.03
SUBSAMPLE_N = 50_000

RESAMPLE_RULE = "1min"
ROLLING_WINDOW_BINS = 10
DEFAULT_HORIZON = "6hr"
HORIZON_OPTIONS = {"1hr": 1, "6hr": 6, "24hr": 24}

FAULT_WINDOWS = [
    ("2020-04-18 00:00:00", "2020-04-18 23:59:00"),
    ("2020-05-29 23:30:00", "2020-05-30 06:00:00"),
    ("2020-06-05 10:00:00", "2020-06-07 14:30:00"),
    ("2020-07-15 14:30:00", "2020-07-15 19:00:00"),
]

ANALOGUE_BASE_FEATURES = [
    "TP2",
    "TP3",
    "H1",
    "DV_pressure",
    "Reservoirs",
    "Oil_temperature",
    "Motor_current",
]

DIGITAL_BASE_FEATURES = [
    "COMP",
    "DV_eletric",
    "Towers",
    "MPG",
    "LPS",
    "Pressure_switch",
    "Oil_level",
    "Caudal_impulses",
]

DERIVED_ANALOGUE_FEATURES = [
    "pressure_drop",
    "pressure_ratio",
    "reservoir_vs_panel",
    "temp_current_product",
    "temp_normalised",
    "load_indicator",
]

DERIVED_DIGITAL_FEATURES = ["compressor_active"]

ANALOGUE_WINDOW_FEATURES = ANALOGUE_BASE_FEATURES + DERIVED_ANALOGUE_FEATURES
DIGITAL_WINDOW_FEATURES = DIGITAL_BASE_FEATURES + DERIVED_DIGITAL_FEATURES

