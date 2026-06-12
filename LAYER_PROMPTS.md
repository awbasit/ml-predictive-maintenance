# Predictive Maintenance System (MetroPT-3) — Layer Prompts & Tracker

## Project overview
- **Thesis topic:** Predictive Maintenance of Machines: A Case Study on UMaT Air Conditions
- **Primary dataset:** MetroPT-3 (UCI) — 1.515M rows, 15 features, real industrial compressor
- **Secondary dataset:** Kaggle Air Compressor — pipeline validation only
- **Target:** Binary fault classification (Normal=0, Fault=1)
- **Label strategy:** 6hr pre-fault prediction horizon (configurable: 1hr, 6hr, 24hr)
- **Models:** Random Forest, Decision Tree, SVM, XGBoost
- **API:** FastAPI on Railway (Docker)
- **Dashboard:** Streamlit on HuggingFace Spaces

---

## Why MetroPT-3 over Kaggle dataset

| Concern | Kaggle (1k rows) | MetroPT-3 (1.5M rows) |
|---|---|---|
| Data origin | Likely synthetic | Real industrial compressor |
| Suspiciously perfect scores | Yes — all models >95% F1 | Expected imperfect, realistic |
| Thesis defensibility | Weak (small, synthetic) | Strong (peer-reviewed, UCI) |
| Time series structure | No | Yes — real temporal dynamics |
| Label quality | Pre-built (suspicious) | Engineered from real fault reports |
| Feature importance honesty | Unclear | Grounded in physics |

---

## Dataset profile (MetroPT-3)

- **Rows:** 1,515,582 at ~1Hz (Feb–Aug 2020)
- **Nulls:** 1 row — dropped
- **Temperature:** Oil_temperature (mean=62°C, fault windows: 73–82°C)
- **Fault windows:** 4 documented air leak events
  - #1: 2020-04-18 (8,657 rows, ~24hrs)
  - #2: 2020-05-29 (2,360 rows, ~6.5hrs)
  - #3: 2020-06-05 (17,315 rows, ~2.5 days)
  - #4: 2020-07-15 (1,622 rows, ~4.5hrs)

---

## Layer prompts

---

### Layer 1 — Data ingestion
**Files:** `src/preprocess.py`, `utils/preprocessing_utils.py` → `load_metropt_data()`

**Prompt:**
```
Load the MetroPT-3 dataset from data/MetroPT3_Dataset.csv.
Parse the timestamp column as datetime. Drop the Unnamed: 0 index column.
Sort by timestamp. Drop the single null row (1 null per column).
The dataset has 1,515,582 rows and 15 sensor columns spanning
Feb–Aug 2020 at approximately 1Hz sampling rate.
Print the date range and row count after loading.
```

---

### Layer 2 — Label engineering
**Files:** `src/preprocess.py`, `utils/preprocessing_utils.py` → `engineer_temporal_labels()`

**Prompt:**
```
Implement temporal label engineering for predictive maintenance.

Four documented fault windows exist (air leak events):
  - 2020-04-18 00:00 to 2020-04-18 23:59
  - 2020-05-29 23:30 to 2020-05-30 06:00
  - 2020-06-05 10:00 to 2020-06-07 14:30
  - 2020-07-15 14:30 to 2020-07-15 19:00

Label strategy:
- label=1 for all rows inside a fault window
- label=1 for all rows within [horizon] before each fault window start
- label=0 for all other rows

Support three configurable horizons: 1hr, 6hr, 24hr.
Default: 6hr (best balance of early warning and precision).

Print the class distribution and fault percentage after labelling.
```

---

### Layer 3 — Feature engineering
**Files:** `src/preprocess.py`, `utils/features.py` → `engineer_row_features()`

**Prompt:**
```
Add derived features to the raw MetroPT-3 sensor data before windowing.

Pressure features:
- pressure_drop = TP3 - TP2  (key degradation indicator)
- pressure_ratio = TP2 / (TP3 + 1e-6)
- reservoir_vs_panel = Reservoirs - TP3

Temperature-current interaction (thesis narrative):
- temp_current_product = Oil_temperature * Motor_current
- temp_normalised = Oil_temperature / (Motor_current + 1e-6)

Operational state features:
- compressor_active = 1 if COMP==0 AND DV_eletric==1 else 0
  (indicates compressor under load based on valve states)
- load_indicator = Motor_current * compressor_active

All features computed on the raw 1Hz timeseries before windowing.
```

---

### Layer 4 — Window aggregation
**Files:** `src/preprocess.py`, `utils/preprocessing_utils.py` → `create_window_features()`

**Prompt:**
```
Aggregate the 1.5M row timeseries into window-based feature vectors.

Strategy:
1. Resample to 1-minute bins using mean aggregation (reduces to ~200K rows)
2. Apply 10-minute rolling window (window=10 bins, min_periods=10)
3. For each analogue feature, compute: mean, std, min, max, range
4. For each digital feature and compressor_active: compute mean
   (proportion of time the signal was active in the window)
5. Label: max of window (1 if any minute in window was fault)

Analogue features to window (5 stats each):
TP2, TP3, H1, DV_pressure, Reservoirs, Oil_temperature, Motor_current,
pressure_drop, pressure_ratio, reservoir_vs_panel,
temp_current_product, temp_normalised, load_indicator

Digital features to window (proportion only):
COMP, DV_eletric, Towers, MPG, LPS, Pressure_switch,
Oil_level, Caudal_impulses, compressor_active

Drop rows with NaN (first 10 minutes of each segment).
Print final windowed dataset shape and class distribution.
```

---

### Layer 5 — Preprocessing pipeline
**Files:** `src/preprocess.py`, `utils/preprocessing_utils.py`, `utils/io_utils.py`

**Prompt:**
```
Build the full preprocessing pipeline for the windowed MetroPT-3 dataset.

Split: 70/15/15 stratified train/val/test (random_state=42, stratify=y).
Scaling: StandardScaler fitted on train only, applied to all splits.
SMOTE: Apply only if class imbalance ratio > 3:1 (k_neighbors=5).

Serialise:
- models/preprocessing_pipeline.pkl (fitted StandardScaler pipeline)
- data/processed/feature_cols.pkl (list of feature column names)
- data/processed/X_train.npy, y_train.npy
- data/processed/X_val.npy, y_val.npy
- data/processed/X_test.npy, y_test.npy

Print split sizes, class distributions, and SMOTE result.
```

---

### Layer 6 — Model training
**Files:** `src/train.py`, `utils/training_utils.py`

**Prompt:**
```
Train 4 models on the windowed MetroPT-3 features with GridSearchCV
(StratifiedKFold n_splits=5, scoring='f1').

1. RandomForestClassifier
   Grid: n_estimators=[100,200,300], max_depth=[10,20,None],
   min_samples_split=[2,5,10], min_samples_leaf=[1,2,4],
   class_weight=['balanced']

2. DecisionTreeClassifier
   Grid: max_depth=[5,10,15,20], min_samples_split=[5,10,20],
   min_samples_leaf=[2,4,8], criterion=['gini','entropy'],
   class_weight=['balanced']

3. SVC (probability=True)
   Grid: C=[0.1,1,10], kernel=['rbf','linear'],
   gamma=['scale','auto'], class_weight=['balanced']

4. XGBClassifier
   Grid: n_estimators=[100,200], max_depth=[3,5,7],
   learning_rate=[0.01,0.05,0.1], subsample=[0.7,0.8,1.0],
   colsample_bytree=[0.7,0.8,1.0], min_child_weight=[1,3,5],
   reg_alpha=[0,0.1,0.5], scale_pos_weight=[imbalance_ratio]

Evaluate on val and test: accuracy, precision, recall, F1, ROC-AUC,
confusion matrix, classification report.
Select best model by validation F1.
Save: models/{name}.pkl, models/best_model.pkl,
models/best_model_meta.pkl, models/evaluation_results.json.
```

---

### Layer 7 — Horizon comparison experiment
**File:** `src/train.py` → `run_horizon_comparison()`
**Run with:** `python src/train.py --mode horizon_compare`

**Prompt:**
```
Run a prediction horizon comparison experiment — core thesis contribution.

For each horizon in [1hr, 6hr, 24hr]:
1. Re-run preprocessing with that horizon key
2. Train a RandomForestClassifier (n_estimators=200, max_depth=20,
   class_weight='balanced', random_state=42)
3. Evaluate on val and test splits

Report for each horizon:
- Number of fault training samples
- Validation F1, Recall, Precision, ROC-AUC
- Test F1, Recall

Print a comparison table and save to models/horizon_comparison.json.

Academic framing: shorter horizons (1hr) give high precision but fewer
warning samples. Longer horizons (24hr) give more warning time but may
include pre-fault data that looks normal, reducing precision.
The 6hr horizon is the thesis recommended default.
```

---

### Layer 8 — FastAPI inference service
**Files:** `api/main.py`, `utils/features.py`, `utils/io_utils.py`

**Prompt:**
```
Build a FastAPI inference service for the MetroPT-3 predictive maintenance model.

On startup: load best_model.pkl, preprocessing_pipeline.pkl,
feature_cols.pkl, best_model_meta.pkl from MODELS_DIR and
PROCESSED_DIR environment variables.

Endpoints:
1. GET / — service info
2. GET /health — model loaded status, name, val_f1
3. POST /predict
   Input: SensorReading with 15 fields matching MetroPT-3 columns:
   TP2, TP3, H1, DV_pressure, Reservoirs, Oil_temperature,
   Motor_current, COMP, DV_eletric, Towers, MPG, LPS,
   Pressure_switch, Oil_level, Caudal_impulses

   Before inference:
   - Apply row-level feature engineering (pressure_drop, pressure_ratio,
     reservoir_vs_panel, temp_current_product, temp_normalised,
     compressor_active, load_indicator)
   - Approximate window features: set mean=reading value,
     std=0, min=value, max=value, range=0 for analogue features;
     prop=value for digital features
   - Scale using loaded pipeline
   - Predict and return probabilities

   Output: PredictionResponse with status (Normal/Fault),
   probability_fault, probability_normal, alert (bool, threshold=0.5),
   alert_message, model_used, horizon

Return 503 if artifacts not loaded, 500 on inference error.
```

---

### Layer 9 — Streamlit dashboard
**Files:** `dashboard/app.py`, `utils/api_client.py`

**Prompt:**
```
Build a Streamlit dashboard for the UMaT AC compressor predictive
maintenance system. Reads API_URL from environment variable.

Layout:
Sidebar:
- API health check (GET /health)
- Shows model name and val F1 as metrics
- About section explaining 6hr prediction horizon

Main area:
Section 1 — Sensor input grouped into 3 subsections:
  (a) Pressure sensors (5 columns): TP2, TP3, H1, DV_pressure, Reservoirs
  (b) Temperature & current (2 columns): Oil_temperature, Motor_current
  (c) Digital valve signals (4 columns using selectbox 0.0/1.0):
      COMP, DV_eletric, Towers, MPG, LPS, Pressure_switch,
      Oil_level, Caudal_impulses

Section 2 — "Run fault prediction" primary button (full width)

Section 3 — Results (shown after prediction):
  - 4 metric cards: status (emoji), fault probability,
    normal probability, model name
  - Error/success banner with alert message
  - Plotly gauge chart (0–100%) with 3 zones:
    green (0–30), amber (30–50), red (50–100)
    with delta reference at 50
  - 2-column sensor snapshot:
    Left: pressure bar chart (TP2, TP3, H1, DV, Reservoirs)
    Right: derived health indicators bar chart
    (pressure_drop, oil_temp, motor_current, temp×current)

Default values: normal operating sample from dataset.
Footer: dataset attribution + framework stack.
Deploy to HuggingFace Spaces (SDK: streamlit).
```

---

### Layer 10 — Docker + deployment
**Files:** `Dockerfile`, `railway.toml`, `dashboard/README.md`

**Prompt:**
```
Containerise the FastAPI predictive maintenance service for Railway
and the Streamlit dashboard for HuggingFace Spaces.

Dockerfile:
- Base: python:3.11-slim, WORKDIR /app
- COPY requirements.txt and pip install --no-cache-dir
- COPY src/, api/, models/, data/processed/, utils/
- ENV MODELS_DIR=models, PROCESSED_DIR=data/processed
- EXPOSE 8000
- CMD: uvicorn api.main:app --host 0.0.0.0 --port 8000

railway.toml:
- builder: DOCKERFILE
- startCommand uses $PORT (Railway injects this at runtime)
- healthcheckPath: /health, timeout: 30s
- restartPolicy: ON_FAILURE, max 3 retries

HuggingFace Spaces (dashboard/README.md):
- YAML front matter: sdk=streamlit, sdk_version=1.37.1, app_file=app.py
- Dashboard reads API_URL from Space secret

Deployment steps:
1. Run: python src/preprocess.py  (generates data/processed/ artifacts)
2. Run: python src/train.py       (generates models/ artifacts)
3. Verify: models/best_model.pkl and models/preprocessing_pipeline.pkl exist
4. Build: docker build -t pm-api .
5. Push to Railway via GitHub or: railway up
6. Set Railway env vars: MODELS_DIR=models, PROCESSED_DIR=data/processed
7. Create HuggingFace Space (Streamlit), push dashboard/ folder contents
8. Add Space secret: API_URL = https://your-app.railway.app
9. Optional: run python src/train.py --mode horizon_compare
   to generate horizon_comparison.json for thesis appendix
```

---

## File structure

```
pm_metropt/
├── data/
│   ├── MetroPT3_Dataset.csv        ← place dataset here
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
│   ├── evaluation_results.json
│   └── horizon_comparison.json    ← after horizon_compare run
│   └── learning_curve_*.png/json  ← generalization diagnostics
├── src/
│   ├── preprocess.py
│   └── train.py
├── utils/
│   ├── __init__.py
│   ├── constants.py
│   ├── features.py
│   ├── preprocessing_utils.py
│   ├── training_utils.py
│   ├── io_utils.py
│   └── api_client.py
├── api/
│   └── main.py
├── dashboard/
│   ├── app.py
│   └── README.md
├── Dockerfile
├── railway.toml
├── requirements.txt
└── LAYER_PROMPTS.md
```

---

## Execution order (run these in sequence)

```bash
# Step 1: place MetroPT3_Dataset.csv in data/
# Step 2:
python src/preprocess.py

# Step 3:
python src/train.py

# Step 4 (optional — for thesis horizon comparison table):
python src/train.py --mode horizon_compare

# Step 5: start API locally to test
uvicorn api.main:app --reload

# Step 6: start dashboard locally
streamlit run dashboard/app.py

# Step 7: deploy
docker build -t pm-api .
railway up
# push dashboard/ to HuggingFace Spaces
```

---

## Day execution log

### Day 1 — Data + Preprocessing: READY TO RUN
- [x] MetroPT-3 profiled: 1.515M rows, 15 features, 4 fault windows confirmed
- [x] Label engineering designed: 6hr pre-fault horizon
- [x] 10 derived features designed
- [x] 10-minute rolling window aggregation strategy designed
- [x] preprocess.py written and ready

### Day 2 — Model Training: READY TO RUN
- [x] All 4 model configs with regularisation grids written
- [x] Horizon comparison experiment designed
- [x] train.py written and ready

### Day 3 — API + Dashboard: COMPLETE
- [x] FastAPI service with /health and /predict
- [x] Streamlit dashboard with gauges and sensor charts

### Day 4 — Deployment: READY
- [x] Dockerfile written
- [x] railway.toml written
- [x] HuggingFace README config written
- [ ] Run preprocess.py on local machine with MetroPT3_Dataset.csv
- [ ] Run train.py on local machine
- [ ] Docker build + Railway deploy
- [ ] HuggingFace Spaces deploy
