# ⚡ ChargeUp — EV Charging Demand Forecast & Fault Detection

Machine-learning capstone based on the **ChargeUp** project brief from the
25SC2107E SLM.

## What the project does

ChargeUp solves four connected problems:

1. **Predict** next-hour EV charging demand in kWh.
2. **Classify** whether a charging station is likely to be faulty.
3. **Segment** stations by usage pattern with K-Means / DBSCAN.
4. **Deploy** the trained system as a FastAPI service with a dashboard,
   health endpoint, metrics, PSI drift monitoring, and a retraining trigger.

The included dataset is **synthetic demo data**, generated only so the project
runs immediately. Replace it with your real charging-session + telemetry +
weather + geography data when available.

## ML models included

### Regression
- Linear Regression baseline
- Ridge Regression
- Random Forest Regressor
- XGBoost Regressor

### Classification
- Logistic Regression
- Decision Tree
- Random Forest Classifier
- XGBoost Classifier

### Unsupervised learning
- K-Means station segments
- DBSCAN station segments
- PCA projection for station structure

### Evaluation
- Regression: MAE, RMSE, R²
- Classification: Precision, Recall, F1, ROC-AUC, PR-AUC
- Temporal train/test split to reduce time leakage
- TimeSeriesSplit cross-validation
- Probability calibration support in the classifier pipeline
- Feature importance / SHAP export for XGBoost
- PSI drift guard

---

## Quick start (Windows PowerShell)

```powershell
cd ChargeUp
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

python -m src.generate_data
python -m src.train

uvicorn api.main:app --reload
```

Then open:

- Dashboard: http://127.0.0.1:8000
- Swagger API: http://127.0.0.1:8000/docs
- Health: http://127.0.0.1:8000/health
- Metrics: http://127.0.0.1:8000/metrics

The ZIP already contains generated demo data and trained artifacts, so after
installing requirements you can normally start directly with:

```powershell
uvicorn api.main:app --reload
```

---

## Linux / macOS

```bash
cd ChargeUp
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn api.main:app --reload
```

---

## API examples

### Combined demand + fault prediction

`POST /predict`

```json
{
  "station_id": "CHG-001",
  "timestamp": "2026-10-13T18:00:00",
  "temperature_c": 31.0,
  "humidity_pct": 64.0,
  "rainfall_mm": 0.0,
  "latitude": 16.5062,
  "longitude": 80.6480,
  "session_count": 7,
  "avg_session_minutes": 43.0,
  "avg_power_kw": 7.2,
  "connector_errors": 0,
  "voltage_v": 230.0,
  "current_a": 28.0,
  "availability_pct": 99.0,
  "energy_kwh": 31.5
}
```

Typical response:

```json
{
  "next_hour_demand_kwh": 34.81,
  "fault_probability": 0.0412,
  "station_fault_flag": 0,
  "fault_label": "NORMAL",
  "station_segment": 2,
  "model_versions": {
    "regression": "...",
    "classification": "..."
  }
}
```

### Drift check

`POST /drift/check`

Send an array of observations. The endpoint calculates feature-level PSI and
returns an overall drift status.

### Retraining trigger

`POST /retrain/trigger`

Creates a local retraining-request marker. In a production system this would
normally enqueue a training job.

---

## Project structure

```text
ChargeUp/
├── api/
│   ├── main.py
│   └── static/
│       ├── index.html
│       ├── app.js
│       └── style.css
├── data/
│   ├── raw/
│   │   └── chargeup_sessions.csv
│   └── processed/
├── models/
│   ├── regression_model.joblib
│   ├── fault_model.joblib
│   ├── station_segments.json
│   ├── station_cluster_models.joblib
│   ├── baseline_stats.json
│   ├── training_metrics.json
│   ├── model_meta.json
│   └── shap_importance.csv
├── notebooks/
│   └── 01_chargeup_walkthrough.ipynb
├── reports/
│   ├── station_segments.csv
│   └── test_predictions.csv
├── scripts/
│   ├── run_dev.ps1
│   └── run_dev.sh
├── src/
│   ├── clustering.py
│   ├── drift.py
│   ├── features.py
│   ├── generate_data.py
│   └── train.py
├── tests/
├── Dockerfile
├── requirements.txt
└── README.md
```

## Important note

The SLM project brief defines the modelling task and deployment expectations,
but does not provide a complete ready-to-download project dataset in the
provided project card. For that reason this repository includes a reproducible
synthetic dataset generator and is structured so a real CSV can replace it.

## Suggested viva one-liner

> ChargeUp forecasts next-hour EV charging demand and flags likely station
> faults using charging-session, telemetry, weather, location, and time
> features; it also clusters stations by usage pattern and serves the models
> through FastAPI with drift monitoring.
