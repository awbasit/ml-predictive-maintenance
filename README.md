# Predictive Maintenance (AC Compressor)

End-to-end binary classification system for bearing fault detection (`Ok` vs `Noisy`) using sensor telemetry.

## Goals

- Keep training and inference feature engineering consistent.
- Keep modules focused and minimal (no dead legacy scripts).
- Make the pipeline reproducible and deployment-ready.

## Project layout

- `src/preprocess.py`: preprocessing orchestration script that uses reusable utilities.
- `src/train.py`: model training orchestration script that uses reusable utilities.
- `api/main.py`: FastAPI service with `/predict`, `/health`, and root endpoints.
- `dashboard/app.py`: Streamlit UI that calls the API over HTTP.
- `utils/`: reusable support modules for constants, IO/artifacts, preprocessing helpers, training helpers, feature engineering, and API client calls.
- `Dockerfile`: containerization for API deployment.

## Data and artifacts

- Input dataset: `data/raw_data.csv`
- Processed outputs: `data/processed/*.npy`, `data/processed/feature_cols.pkl`
- Model outputs: `models/*.pkl`, `models/evaluation_results.json`
- Generalization diagnostics: `models/learning_curve_<best_model>.png`, `models/learning_curve_<best_model>.json`

## Quick start

1. Install dependencies:
   - `pip install -r requirements.txt`
2. Run preprocessing:
   - `python src/preprocess.py`
3. Train models:
   - `python src/train.py`
4. Start API:
   - `uvicorn api.main:app --host 0.0.0.0 --port 8000`
5. Start dashboard (optional, in a new shell):
   - `streamlit run dashboard/app.py`

## API contract

- `GET /health`: service and artifact readiness.
- `POST /predict`: accepts 20 raw sensor fields and returns:
  - predicted status (`Ok` or `Noisy`)
  - probabilities
  - alert boolean and message
  - active model name

## Engineering standards used

- Single source of truth for derived features (`utils/features.py`).
- No training/inference duplication for transformations.
- Artifact paths controlled through `MODELS_DIR` and `PROCESSED_DIR`.
- Supporting functions are centralized in `utils/` and reused across modules.
- Modules are separated by responsibility (preprocess, train, serve, dashboard).
- Training adds Gaussian noise augmentation and learning-curve tracking to detect and reduce memorization risk.
