# Predictive Maintenance System — Layer Prompts & Tracker

## Project overview
- Dataset: Kaggle Air Compressor (1,000 rows, 26 columns)
- Target: `bearings` (binary — Ok / Noisy)
- Models: Random Forest, Decision Tree, SVM, XGBoost
- API: FastAPI on Railway (Docker)
- Dashboard: Streamlit on HuggingFace Spaces
- Validation dataset: MetroPT-3 (UCI)
- Shared feature engineering module: `utils/features.py` (single source of truth)

---

## Results summary (actual training run)

| Model | Val Accuracy | Val F1 | Val ROC-AUC | Test F1 |
|---|---|---|---|---|
| Random Forest | 0.9800 | 0.9508 | 0.9975 | 0.8615 |
| Decision Tree | 0.9933 | 0.9831 | 0.9833 | 0.9508 |
| SVM | **1.0000** | **1.0000** | **1.0000** | 0.9508 |
| XGBoost | 0.9933 | 0.9836 | 1.0000 | 0.9508 |

Best model: **SVM** (C=10, kernel=linear, gamma=scale)

---

## Layer prompts

These are the detailed prompts used to build each layer. Use them to
regenerate, extend, or debug any layer independently.

---

### Layer 1 — Data ingestion
**File:** `src/preprocess.py` (load_data, encode_target sections)

**Prompt:**
```
You are building a data ingestion module for a predictive maintenance
system for air conditioning compressors. Load the Kaggle air compressor
dataset (data/raw_data.csv). The dataset has 1,000 rows and 26 columns.
The target column is 'bearings' with values 'Ok' and 'Noisy'.
Drop these columns entirely: id, wpump, radiator, exvalve, acmotor
(acmotor is constant — all 'Stable' — zero predictive value).
Encode the target: Noisy=1, Ok=0. Print the class distribution after
encoding. Return a clean DataFrame ready for feature engineering.
```

---

### Layer 2 — Preprocessing pipeline
**Files:** `src/preprocess.py`, `utils/features.py`

**Prompt:**
```
You are building a preprocessing pipeline for an air compressor
predictive maintenance model. The dataset has 30 feature columns after
dropping irrelevant ones.

Feature engineering steps to apply:
1. Temperature differentials (core thesis narrative):
   - temp_water_delta = water_outlet_temp - water_inlet_temp
   - temp_outlet_vs_oil = outlet_temp - oil_tank_temp
   - temp_mean = mean of [outlet_temp, water_inlet_temp, water_outlet_temp, oil_tank_temp]
   - temp_max = max of the 4 temperature features
2. Mechanical load ratios:
   - power_per_rpm = motor_power / (rpm + 1)
   - torque_per_rpm = torque / (rpm + 1)
   - pressure_air_ratio = outlet_pressure_bar / (air_flow + 1)
3. Vibration magnitudes:
   - vib_ground_magnitude = sqrt(gaccx^2 + gaccy^2 + gaccz^2)
   - vib_head_magnitude = sqrt(haccx^2 + haccy^2 + haccz^2)
   - vib_ratio = vib_head_magnitude / (vib_ground_magnitude + 1e-6)

Split strategy: 70/15/15 stratified train/val/test split (random_state=42).
Scaling: StandardScaler fitted on train only, applied to all splits.
Class imbalance: Apply SMOTE (k_neighbors=5, random_state=42) to training
split only — never to val or test splits.
Serialise: Save the fitted pipeline as models/preprocessing_pipeline.pkl.
Save feature column names as data/processed/feature_cols.pkl.
Save all splits as .npy files in data/processed/.
```

---

### Layer 3 — Model training & evaluation
**File:** `src/train.py`

**Prompt:**
```
You are building a multi-model training and evaluation pipeline for
binary classification (bearing fault detection).

Train these 4 models with GridSearchCV (StratifiedKFold, n_splits=5,
scoring='f1'):

1. RandomForestClassifier
   - Grid: n_estimators=[100,200], max_depth=[None,10,20],
     min_samples_split=[2,5], class_weight=['balanced']

2. DecisionTreeClassifier
   - Grid: max_depth=[5,10,20,None], min_samples_split=[2,5,10],
     criterion=['gini','entropy'], class_weight=['balanced']

3. SVC (with probability=True)
   - Grid: C=[0.1,1,10], kernel=['rbf','linear'],
     gamma=['scale','auto'], class_weight=['balanced']

4. XGBClassifier
   - Grid: n_estimators=[100,200], max_depth=[3,6,9],
     learning_rate=[0.01,0.1], subsample=[0.8,1.0],
     scale_pos_weight=[4]  # accounts for 80/20 class imbalance

Evaluation metrics for each model on both val and test splits:
accuracy, F1 (binary), ROC-AUC, confusion matrix, classification report.

Select best model by validation F1. Save it as models/best_model.pkl.
Save all individual models as models/{name}.pkl.
Save evaluation results as models/evaluation_results.json.
Save best model metadata (name, val_f1) as models/best_model_meta.pkl.
Print a comparison table at the end.
```

---

### Layer 4 — FastAPI inference service
**Files:** `api/main.py`, `utils/features.py`

**Prompt:**
```
You are building a production FastAPI inference service for an AC
compressor predictive maintenance system.

On startup: load models/best_model.pkl, models/preprocessing_pipeline.pkl,
data/processed/feature_cols.pkl, and models/best_model_meta.pkl.
Use environment variables MODELS_DIR and PROCESSED_DIR with defaults.

Endpoints:
1. GET /health
   Returns: status (healthy/degraded), model_loaded (bool),
   model_name, model_val_f1

2. POST /predict
   Input: SensorReading Pydantic model with 20 numerical fields
   matching these columns: rpm, motor_power, torque,
   outlet_pressure_bar, air_flow, noise_db, outlet_temp,
   wpump_outlet_press, water_inlet_temp, water_outlet_temp,
   wpump_power, water_flow, oilpump_power, oil_tank_temp,
   gaccx, gaccy, gaccz, haccx, haccy, haccz.

   Before scaling, apply the same feature engineering used in training:
   temp_water_delta, temp_outlet_vs_oil, temp_mean, temp_max,
   power_per_rpm, torque_per_rpm, pressure_air_ratio,
   vib_ground_magnitude, vib_head_magnitude, vib_ratio.

   Output PredictionResponse: status (Ok/Noisy), probability_noisy,
   probability_ok, alert (bool, threshold=0.5), alert_message,
   model_used.

3. GET / — root info endpoint

Use Pydantic v2. Return 503 if model not loaded, 500 on inference error.
```

---

### Layer 5a — Streamlit dashboard
**File:** `dashboard/app.py`

**Prompt:**
```
You are building a Streamlit dashboard for an AC compressor predictive
maintenance system. The dashboard calls a FastAPI backend via HTTP.

API_URL is read from environment variable with default http://localhost:8000.

Layout:
- Sidebar: API health check (GET /health), shows model name and val F1
  as metrics, shows error if unreachable.
- Main area:
  - Section 1: Sensor input form grouped into 3 subsections:
    (a) Temperature sensors: outlet_temp, water_inlet_temp,
        water_outlet_temp, oil_tank_temp — 4 columns, these are the
        primary thesis narrative features.
    (b) Mechanical sensors: rpm, motor_power, torque,
        outlet_pressure_bar, air_flow, noise_db, wpump_outlet_press,
        wpump_power, water_flow, oilpump_power.
    (c) Vibration sensors: gaccx, gaccy, gaccz, haccx, haccy, haccz.
  - Section 2: "Run prediction" primary button (full width).
  - Section 3 (after prediction):
    - 3 metric cards: bearing status (with red/green emoji), fault
      probability, model name.
    - Success/error banner with alert message.
    - Plotly gauge chart (0-100%) showing fault probability with
      green zone 0-50, red zone 50-100, red threshold line at 50.
    - Plotly bar chart of the 4 temperature readings with a warm
      colour scale (green -> amber -> red).

Use default sensor values matching a normal operating sample from the
dataset. Deploy to HuggingFace Spaces (SDK: streamlit).
```

---

### Layer 6 — Docker + deployment
**Files:** `Dockerfile`, `railway.toml`, `dashboard/README.md`

**Prompt:**
```
You are containerising a FastAPI predictive maintenance service for
Railway deployment and a Streamlit dashboard for HuggingFace Spaces.

Dockerfile (FastAPI on Railway):
- Base: python:3.11-slim
- WORKDIR /app
- Copy and install requirements.txt first (layer caching)
- Copy src/, api/, models/, data/processed/ into image
- Set ENV MODELS_DIR=models and PROCESSED_DIR=data/processed
- EXPOSE 8000
- CMD: uvicorn api.main:app --host 0.0.0.0 --port 8000

railway.toml:
- builder: DOCKERFILE
- startCommand uses $PORT env variable (Railway injects this)
- healthcheckPath: /health
- healthcheckTimeout: 30s
- restartPolicy: ON_FAILURE, max 3 retries

HuggingFace Spaces (Streamlit dashboard):
- dashboard/README.md must contain the Spaces YAML front matter:
  sdk: streamlit, sdk_version: 1.37.1, app_file: app.py
- The dashboard reads API_URL from environment variable so Railway
  URL can be injected as a Space secret.
- requirements.txt for the Spaces repo: streamlit, plotly, requests

Deployment steps:
1. Train models locally, verify artifacts exist in models/ and data/processed/
2. Build Docker image: docker build -t predictive-maintenance-api .
3. Push to Railway via GitHub or Railway CLI
4. Set Railway env vars: MODELS_DIR, PROCESSED_DIR
5. Create HuggingFace Space (Streamlit SDK), push dashboard/
6. Set API_URL as a Space secret pointing to Railway public URL
```

---

## Project file structure

```
predictive_maintenance/
├── README.md
├── data/
│   ├── raw_data.csv
│   └── processed/
│       ├── X_train.npy
│       ├── y_train.npy
│       ├── X_val.npy
│       ├── y_val.npy
│       ├── X_test.npy
│       ├── y_test.npy
│       └── feature_cols.pkl
├── models/
│   ├── preprocessing_pipeline.pkl
│   ├── random_forest.pkl
│   ├── decision_tree.pkl
│   ├── svm.pkl
│   ├── xgboost.pkl
│   ├── best_model.pkl
│   ├── best_model_meta.pkl
│   └── evaluation_results.json
├── src/
│   ├── preprocess.py
│   └── train.py
├── utils/
│   ├── __init__.py
│   ├── api_client.py
│   ├── constants.py
│   ├── features.py
│   ├── io_utils.py
│   ├── preprocessing_utils.py
│   └── training_utils.py
├── api/
│   └── main.py
├── dashboard/
│   ├── app.py
│   └── README.md  (HF Spaces config)
├── Dockerfile
├── railway.toml
└── requirements.txt
```

---

## Audit hardening log

- [x] Removed irrelevant legacy script (`maintenance.py`)
- [x] Centralised feature engineering in `utils/features.py`
- [x] Reused shared feature engineering in both preprocessing and API inference
- [x] Added reproducible runtime dependencies in `requirements.txt`
- [x] Added `README.md` for architecture/run/deployment documentation
- [x] Added deployment metadata files (`railway.toml`, `dashboard/README.md`)

---

## Day-by-day execution log

### Day 1 status: COMPLETE
- [x] Dataset loaded and profiled (1,000 rows, 26 cols, 0 missing)
- [x] Target encoded (0=Ok, 1=Noisy)
- [x] acmotor dropped (constant feature)
- [x] 10 engineered features added (temperature diffs, mechanical ratios, vibration magnitudes)
- [x] StandardScaler pipeline fitted and serialised
- [x] SMOTE applied — 700 -> 1,120 balanced training samples
- [x] 70/15/15 stratified splits saved as .npy

### Day 2 status: COMPLETE
- [x] Random Forest trained — Val F1: 0.9508
- [x] Decision Tree trained — Val F1: 0.9831
- [x] SVM trained — Val F1: 1.0000 (BEST)
- [x] XGBoost trained — Val F1: 0.9836
- [x] All models serialised to models/
- [x] evaluation_results.json saved

### Day 3 status: COMPLETE
- [x] FastAPI service built (main.py)
- [x] /health and /predict endpoints implemented
- [x] Pydantic input validation
- [x] Streamlit dashboard built (app.py)
- [x] Gauge chart + temperature bar chart
- [x] HuggingFace Spaces README config

### Day 4 status: READY
- [x] Dockerfile written
- [x] railway.toml written
- [ ] Deploy FastAPI to Railway (requires Railway account + CLI)
- [ ] Deploy dashboard to HuggingFace Spaces (requires HF account)
- [ ] Set API_URL secret in HF Spaces
