import sys
from pathlib import Path
import joblib
import pandas as pd

# Allow running as script: `python src/train.py`
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

model = joblib.load("models/xgboost.pkl")
feature_cols = joblib.load("data/processed/feature_cols.pkl")

importance = pd.Series(model.feature_importances_, index=feature_cols)
top15 = importance.nlargest(15).sort_values()
for name, val in top15.items():
    print(f"{name}: {val:.6f}")