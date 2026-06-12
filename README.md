# Predictive Maintenance System (MetroPT-3)

Refreshed end-to-end predictive maintenance pipeline for UMaT air-compressor thesis work, built on the MetroPT-3 real industrial dataset.

## What This Version Targets

- Realistic fault detection using MetroPT-3 (not toy/synthetic behavior).
- Configurable temporal labels (`1hr`, `6hr`, `24hr`) with default `6hr`.
- Reusable utilities aligned with the latest core modules:
  - `src/preprocess.py`
  - `src/train.py`
  - `api/main.py`
  - `dashboard/app.py`

## Architecture

- `src/preprocess.py`: load dataset, engineer labels/features, build rolling windows, split/scale/SMOTE, save artifacts.
- `src/train.py`: train RF/DT/SVM/XGBoost with regularization, add train noise augmentation, save best model, export learning curves, run horizon comparison.
- `api/main.py`: serve `/predict` and `/health`, approximate window features from one reading.
- `dashboard/app.py`: Streamlit interface for MetroPT-3 sensor inputs and visual prediction results.
- `utils/`: shared constants, feature engineering, preprocessing helpers, training helpers, IO helpers, API client helpers.

## Expected Data and Artifacts

- Input dataset: `data/MetroPT3_Dataset.csv`
- Preprocessing artifacts: `data/processed/*.npy`, `data/processed/feature_cols.pkl`
- Model artifacts: `models/*.pkl`, `models/evaluation_results.json`
- Diagnostics: `models/learning_curve_<best_model>.png`, `models/learning_curve_<best_model>.json`
- Optional experiment: `models/horizon_comparison.json`

## Refresh Start (Clean Run)

From project root:

1. Create and activate virtual environment (PowerShell)
   - `python -m venv .venv`
   - `.\.venv\Scripts\Activate.ps1`
2. Install dependencies
   - `pip install -r requirements.txt`
3. Confirm dataset exists at `data/MetroPT3_Dataset.csv`
4. Run preprocessing (`6hr` default)
   - `python src/preprocess.py`
5. Train models
   - `python src/train.py`
6. Optional horizon experiment
   - `python src/train.py --mode horizon_compare`
7. Start API (terminal 1)
   - `uvicorn api.main:app --host 0.0.0.0 --port 8000`
8. Start dashboard (terminal 2)
   - `streamlit run dashboard/app.py`

After startup:
- API docs: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/health`
- Dashboard UI: `http://localhost:8501`

## Docker (API)

- Build: `docker build -t pm-api .`
- Run: `docker run --rm -p 8000:8000 pm-api`

## Environment Variables

- `MODELS_DIR` (default: `models`)
- `PROCESSED_DIR` (default: `data/processed`)
- `API_URL` for dashboard (default: `http://localhost:8000`)
