# Compressor Predictive Maintenance

Real-time fault detection for industrial air compressors using the **MetroPT-3** dataset.  
A full-stack ML system: preprocessing pipeline → ensemble models → FastAPI inference service → Streamlit HMI dashboard.

---

## Overview

This project implements a predictive maintenance pipeline for the MetroPT-3 metro air compressor dataset. It labels sensor windows with **three prediction horizons** (1 hr, 6 hr, 24 hr before fault onset), trains four tree-based classifiers, and serves predictions through a REST API with a live HMI-style dashboard.

```
Raw sensor data (1.5 M rows, 15 Hz)
        │
        ▼
┌───────────────────────┐
│  Layer 1 — Ingestion  │  Resample → 1-min intervals → row-level derived features
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│  Layer 2 — Windows    │  10-min rolling window, 1-min step → mean/std/min/max/range
│                       │  ≈ 100 K feature vectors
│                       │  Label: 1 if within [1hr / 6hr / 24hr] of a fault window
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│  Layer 3 — Training   │  Decision Tree (baseline)
│                       │  Random Forest  (fixed params)
│                       │  Extra Trees    (RandomizedSearchCV on 50 K subsample)
│                       │  XGBoost        (RandomizedSearchCV on 50 K subsample)
│                       │  class_weight="balanced" — no SMOTE
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐       ┌──────────────────────────┐
│  Layer 4 — FastAPI    │◄─────►│  Layer 5 — Dashboard     │
│  /predict  /metrics   │  HTTP │  Live sensor inputs      │
│  /feature-importance  │       │  Donut gauge + radar      │
│  /health   /history   │       │  Model performance tab    │
└───────────────────────┘       └──────────────────────────┘
```

---

## Model Performance (6-hour horizon, validation set)

| Model | F1 | Recall | Precision | ROC AUC |
|---|---|---|---|---|
| Decision Tree | 0.6432 | 0.9741 | 0.4801 | 0.9863 |
| Random Forest | 0.9290 | 0.9565 | 0.9031 | 0.9991 |
| Extra Trees | 0.9260 | 0.9212 | 0.9309 | 0.9963 |
| **XGBoost** ✓ | **0.9517** | **0.9699** | **0.9341** | **0.9998** |

> XGBoost is selected as the production model. The **2.5% fault rate** is handled via `class_weight="balanced"` / `scale_pos_weight` — no SMOTE.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Data / ML | Python 3.11, pandas, numpy, scikit-learn, XGBoost, joblib |
| API | FastAPI 0.136, Uvicorn, Pydantic v2 |
| Dashboard | Streamlit 1.58, Plotly 6 |
| Containerisation | Docker (python:3.11-slim) |
| API deployment | [Railway](https://railway.app) |
| Dashboard deployment | [Hugging Face Spaces](https://huggingface.co/spaces) |

---

## Project Structure

```
predictive-maintenance/
├── api/
│   ├── main.py                  # FastAPI service (predict, metrics, health)
│   └── requirements.txt         # API-only dependencies (Docker)
├── dashboard/
│   ├── app.py                   # Streamlit HMI dashboard
│   ├── requirements.txt         # Dashboard-only dependencies (HF Spaces)
│   └── README.md                # HuggingFace Space metadata
├── src/
│   ├── preprocess.py            # Preprocessing orchestrator
│   └── train.py                 # Training orchestrator
├── utils/
│   ├── api_client.py            # HTTP helpers for dashboard ↔ API
│   ├── constants.py             # Project-wide config & feature lists
│   ├── features.py              # Inference-time feature approximation
│   ├── io_utils.py              # File I/O helpers
│   ├── preprocessing_utils.py   # Window features, label engineering
│   └── training_utils.py        # Model builders, search configs
├── models/                      # Trained model artifacts (committed)
│   ├── best_model.pkl
│   ├── best_model_meta.pkl
│   ├── preprocessing_pipeline.pkl
│   ├── evaluation_results.json
│   └── *.pkl                    # Per-model checkpoints
├── data/
│   └── processed/
│       └── feature_cols.pkl     # Feature column list for inference
├── notebook/
│   └── experiments.ipynb        # Interactive pipeline notebook
├── .streamlit/
│   └── config.toml              # Streamlit dark theme
├── Dockerfile                   # API container
├── railway.toml                 # Railway deployment config
└── requirements.txt             # Full dev environment
```

---

## Local Setup

### Prerequisites

- Python 3.11+
- Git

### 1. Clone and create virtual environment

```bash
git clone https://github.com/<your-username>/predictive-maintenance.git
cd predictive-maintenance

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. (Optional) Re-run preprocessing

Only needed if you have the raw `data/MetroPT3_Dataset.csv` and want to regenerate splits.

```bash
python src/preprocess.py
```

### 4. (Optional) Re-train models

```bash
python src/train.py
```

### 5. Start the API

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

API docs available at `http://localhost:8000/docs`.

### 6. Start the dashboard

In a second terminal:

```bash
streamlit run dashboard/app.py
```

Dashboard available at `http://localhost:8501`.

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `API_URL` | `http://localhost:8000` | Base URL of the FastAPI service |
| `MODELS_DIR` | `models` | Directory containing model `.pkl` files |
| `PROCESSED_DIR` | `data/processed` | Directory containing `feature_cols.pkl` |

---

## Deploying the API to Railway

### Prerequisites

- A [Railway](https://railway.app) account
- The repository pushed to GitHub (models **must** be committed — do not gitignore `models/`)

### Steps

1. **Push your repository to GitHub** (including the `models/` and `data/processed/feature_cols.pkl` files).

2. **Create a new Railway project**  
   Go to [railway.app/new](https://railway.app/new) → **Deploy from GitHub repo** → select your repository.

3. **Railway auto-detects `railway.toml`**  
   The included `railway.toml` configures:
   - Builder: `DOCKERFILE` (uses the root `Dockerfile`)
   - Start command: `uvicorn api.main:app --host 0.0.0.0 --port $PORT`
   - Health check: `GET /health`

4. **Add environment variables** (optional overrides)  
   In Railway → your service → **Variables**:
   ```
   MODELS_DIR=models
   PROCESSED_DIR=data/processed
   ```
   `PORT` is set automatically by Railway.

5. **Deploy**  
   Railway builds the Docker image and deploys. The first build takes 3–5 minutes.

6. **Copy the public URL**  
   Railway assigns a URL like `https://predictive-maintenance-api.up.railway.app`.  
   Test it:
   ```bash
   curl https://<your-app>.up.railway.app/health
   ```

> **Tip:** If you update the models, just push to GitHub — Railway redeploys automatically on every push to `main`.

---

## Deploying the Dashboard to Hugging Face Spaces

### Prerequisites

- A [Hugging Face](https://huggingface.co) account
- The Railway API URL from the step above

### Steps

1. **Create a new Space**  
   Go to [huggingface.co/new-space](https://huggingface.co/new-space):
   - **Space name:** `compressor-predictive-maintenance` (or any name)
   - **SDK:** Streamlit
   - **Visibility:** Public (or Private)
   - Click **Create Space**

2. **Clone the Space repository locally**

   ```bash
   git clone https://huggingface.co/spaces/<your-username>/compressor-predictive-maintenance hf-space
   cd hf-space
   ```

3. **Copy the required files into the Space repo**

   ```bash
   # From your project root
   cp dashboard/app.py         hf-space/
   cp dashboard/requirements.txt hf-space/
   cp dashboard/README.md      hf-space/   # contains the HF --- metadata block
   cp -r .streamlit/           hf-space/.streamlit/
   mkdir -p hf-space/utils
   cp utils/api_client.py      hf-space/utils/
   ```

   Your Space repo should look like:
   ```
   hf-space/
   ├── README.md               # must start with the --- YAML block
   ├── app.py
   ├── requirements.txt
   ├── .streamlit/
   │   └── config.toml
   └── utils/
       └── api_client.py
   ```

4. **Set the `API_URL` Space secret**  
   In HuggingFace → your Space → **Settings** → **Repository secrets**:
   ```
   API_URL = https://<your-app>.up.railway.app
   ```

5. **Push to the Space**

   ```bash
   cd hf-space
   git add .
   git commit -m "deploy dashboard"
   git push
   ```

   HuggingFace will build and start the Space automatically (≈ 2 minutes).

6. **Open your Space**  
   Visit `https://huggingface.co/spaces/<your-username>/compressor-predictive-maintenance`.

> **Note:** Because the dashboard reads `evaluation_results.json` directly from disk as an offline fallback, you only need the API running for live predictions. Model performance charts load from disk automatically.

---

## Dataset

**MetroPT-3** — Predictive Maintenance Dataset for a Metro Air Compressor  
Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/791/metropt+3+dataset)

| Property | Value |
|---|---|
| Raw samples | ~1.5 million rows at 1 Hz |
| Sensors | 7 analogue + 8 digital |
| Documented faults | 4 fault windows (2020) |
| After 10-min windowing | ~100 K feature vectors |
| Fault rate | ~2.5% |
| Prediction horizons | 1 hr / 6 hr / 24 hr |

---

## License

MIT — see [LICENSE](LICENSE) if present.
