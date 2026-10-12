# ⚡ ChargeUp — EV Charging Demand Forecast & Fault Detection

Machine-learning capstone implemented according to the **ChargeUp** project card in the 25SC2107E SLM.

## Project goals

ChargeUp combines four connected tasks:

1. forecast **next-hour EV charging demand (kWh)**
2. classify **station fault / normal state**
3. cluster stations by usage pattern using **K-Means, DBSCAN and PCA**
4. deploy the champion models with **FastAPI, health/metrics, PSI drift monitoring and a retraining trigger**

The SLM card does not provide a complete downloadable raw dataset, so this repository contains a reproducible synthetic charging-session generator with time, weather, geography, charging and telemetry fields. A real dataset can replace it later without changing the project structure.

## SLM 12-week implementation

| Week | SLM work implemented | Main artifacts |
|---|---|---|
| 1 | targets + initial EDA | `src/week01_eda.py`, `docs/week01_*` |
| 2 | ingestion, cleaning, missing values, outliers, leakage checks | `src/week02_cleaning.py`, `reports/week02_data_quality.json` |
| 3 | OLS + from-scratch Gradient Descent + MLflow hook | `src/week03_linear_regression.py`, `reports/week03_linear_regression_metrics.json` |
| 4 | Ridge, Lasso, Logistic Regression + engineered time features | `src/week04_regularized_models.py` |
| 5 | TimeSeriesSplit CV + one-standard-error rule | `src/week05_cv_selection.py` |
| 6 | interpretable Decision Tree + top splits | `src/week06_decision_tree.py` |
| 7 | Random Forest + OOB error + permutation importance | `src/week07_random_forest.py` |
| 8 | XGBoost + early stopping + SHAP | `src/week08_xgboost_shap.py` |
| 9 | K-Means + DBSCAN + PCA + segment feedback feature | `src/week09_station_clustering.py` |
| 10 | nested CV + calibration + imbalance metrics + significance tests | `src/week10_nested_validation.py` |
| 11 | versioned champion package + shared feature module + PSI baseline | `src/week11_package_champion.py`, `models/registry/chargeup-v1.0.0/` |
| 12 | containerized FastAPI + health/metrics/drift/retraining | `api/main.py`, `Dockerfile`, `scripts/start_production.sh` |

Detailed completion notes are under `docs/weekXX_completion.md`.

## Verified results

Using the reproducible 5,360-row demo dataset with 16 stations:

- Week 3 OLS RMSE: **5.922**, R²: **0.705**
- Week 8 XGBoost demand RMSE: **5.090**, R²: **0.782**
- Week 10 nested-validation XGBoost demand RMSE: **4.829**, R²: **0.804**
- Week 10 Logistic fault PR-AUC: **0.938**
- Week 10 calibrated XGBoost fault F1: **0.870**
- Week 9 PCA first two components explain about **78%** of station-level variance
- local API/unit verification: **4 pytest tests passed**

## Quick start — Windows PowerShell

```powershell
git clone https://github.com/void-tech-shiv/chargeUP.git
cd chargeUP
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

python -m src.generate_data
python -m src.week02_cleaning
python -m src.week11_package_champion

uvicorn api.main:app --reload
```

Open:

- Dashboard: `http://127.0.0.1:8000`
- Swagger API: `http://127.0.0.1:8000/docs`
- Health: `http://127.0.0.1:8000/health`
- Metrics: `http://127.0.0.1:8000/metrics`

## Run the complete SLM pipeline

```powershell
python -m src.generate_data
python -m src.week02_cleaning
python -m src.week03_linear_regression
python -m src.week04_regularized_models
python -m src.week05_cv_selection
python -m src.week06_decision_tree
python -m src.week07_random_forest
python -m src.week08_xgboost_shap
python -m src.week09_station_clustering
python -m src.week10_nested_validation
python -m src.week11_package_champion
python -m pytest -q
uvicorn api.main:app --reload
```

Week 1 is EDA/target definition and is reproducible with:

```powershell
python -m src.week01_eda
```

MLflow is included in `requirements.txt`; Week 3 creates the first MLflow experiment/run automatically when the package is installed.

## Docker

The Docker startup flow creates any missing demo data, performs Week 2 cleaning, builds the Week 11 versioned model package, and then starts FastAPI.

```bash
docker build -t chargeup .
docker run --rm -p 8000:8000 chargeup
```

## API endpoints

- `POST /predict` — demand forecast + fault alert
- `POST /predict/demand` — next-hour demand only
- `POST /predict/fault` — fault probability / flag
- `GET /health` — service and model health
- `GET /metrics` — runtime counters + validation/model metadata
- `POST /drift/check` — PSI drift calculation
- `POST /retrain/trigger` — records a retraining request
- `GET /demo/stations` — demo predictions from latest station rows

Example request:

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

## Repository structure

```text
ChargeUp/
├── api/
│   ├── main.py
│   └── static/
├── data/
│   ├── raw/
│   └── processed/
├── docs/
│   └── week01...week12 documentation
├── models/
│   └── registry/chargeup-v1.0.0/
├── reports/
│   └── week02...week12 evaluation outputs
├── scripts/
│   ├── run_dev.ps1
│   ├── run_dev.sh
│   └── start_production.sh
├── src/
│   ├── generate_data.py
│   ├── week01_eda.py
│   ├── week02_cleaning.py
│   ├── week03_linear_regression.py
│   ├── week04_regularized_models.py
│   ├── week05_cv_selection.py
│   ├── week06_decision_tree.py
│   ├── week07_random_forest.py
│   ├── week08_xgboost_shap.py
│   ├── week09_station_clustering.py
│   ├── week10_nested_validation.py
│   ├── model_features.py
│   └── week11_package_champion.py
├── tests/
├── Dockerfile
├── requirements.txt
└── README.md
```

## Viva one-liner

> ChargeUp forecasts next-hour EV charging demand and detects likely charging-station faults from charging, telemetry, weather, geography and time features; it also clusters stations by usage pattern and deploys versioned models through FastAPI with PSI drift monitoring and a retraining trigger.
