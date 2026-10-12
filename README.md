# ⚡ ChargeUp — Palo Alto EV Charging Demand & Fault-Proxy Forecasting

ChargeUp is the 25SC2107E SLM capstone implemented with the **real City of Palo Alto Electric Vehicle Charging Station Usage (July 2011–Dec 2020) dataset**.

Official dataset page:
https://data.paloalto.gov/datasets/194693/electric-vehicle-charging-station-usage-july-2011-dec-2020/

The old `data.cityofpaloalto.org` link can return 404. `src/palo_alto_data.py` first tries the current City download endpoint and then uses a public research mirror of the same `ChargePoint Data CY20Q4.csv` file if needed.

## What the real-data project does

1. Aggregates ChargePoint sessions into **station-hour** observations.
2. Forecasts the station's **next exact-hour charging demand (kWh)**.
3. Predicts a **next-hour ChargePoint server-termination proxy** for the SLM fault-classification task.
4. Segments stations with K-Means / DBSCAN and PCA.
5. Runs OLS, gradient descent, Ridge, Lasso, Logistic Regression, Decision Tree, Random Forest and XGBoost.
6. Uses temporal CV, one-standard-error selection, nested CV, calibration, imbalance-aware metrics, significance tests and SHAP.
7. Packages a versioned `chargeup-paloalto-v2.0.0` model and serves it through FastAPI with health, metrics, PSI drift and retraining endpoints.

## Important label note

The provided City CSV **does not contain a ground-truth hardware-failure / broken-station label**. It does contain `Ended By`, including records such as `CPS Server`. To avoid inventing fault labels, ChargeUp defines `station_fault_flag` as a transparent weak-supervision target: **whether the next exact hour at the same station contains a server-ended charging session**. This is a server-termination risk proxy, not proof of broken hardware.

The City CSV also does **not** contain weather, voltage, current or hardware-availability telemetry. Those values are not fabricated. The real-data model uses charging-session, time, port, fee and geography features available in the source.

## Real source columns used

- Station Name
- Start Date / End Date
- Total Duration / Charging Time
- Energy (kWh)
- Port Type / Port Number
- Latitude / Longitude
- Fee
- Ended By

The preprocessing script derives station-hour features such as session count, average charging duration, average power, hourly energy, fee totals and port-type shares.

## One-command run

### Windows PowerShell

```powershell
git clone https://github.com/void-tech-shiv/chargeUP.git
cd chargeUP
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m src.run_palo_alto_pipeline
uvicorn api.main:app --reload
```

### Faster real-data demo

The official portal export is capped at 10,000 rows. For a faster demo using real Palo Alto rows:

```powershell
python -m src.run_palo_alto_pipeline --export-only --no-mlflow
```

For the full project, run without `--export-only`.

## Manual SLM sequence

```bash
python -m src.palo_alto_data --download
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
```

## FastAPI

After Week 11 artifacts exist:

```bash
uvicorn api.main:app --reload
```

Open:

- Dashboard: `http://127.0.0.1:8000`
- Swagger: `http://127.0.0.1:8000/docs`
- Health: `http://127.0.0.1:8000/health`
- Metrics: `http://127.0.0.1:8000/metrics`

Main endpoints:

- `POST /predict`
- `POST /predict/demand`
- `POST /predict/fault`
- `POST /drift/check`
- `POST /retrain/trigger`

Example input:

```json
{
  "station_id": "PALO ALTO CA / HAMILTON #1",
  "timestamp": "2020-12-01T18:00:00",
  "latitude": 37.444572,
  "longitude": -122.160309,
  "session_count": 2,
  "avg_session_minutes": 75.0,
  "avg_total_duration_minutes": 110.0,
  "avg_power_kw": 6.1,
  "energy_kwh": 14.2,
  "fee_usd": 0.0,
  "level1_share": 0.0,
  "level2_share": 1.0,
  "dc_fast_share": 0.0
}
```

## SLM mapping

| Week | Deliverable |
|---|---|
| 1 | Real source / targets / initial EDA |
| 2 | Palo Alto ingestion, cleaning, hourly aggregation, leakage-safe targets |
| 3 | OLS + from-scratch Gradient Descent + MLflow |
| 4 | Ridge + Lasso + Logistic Regression + time feature engineering |
| 5 | Time-series CV + one-standard-error rule |
| 6 | Interpretable Decision Tree |
| 7 | Random Forest + OOB + permutation importance |
| 8 | XGBoost + early stopping + SHAP |
| 9 | K-Means + DBSCAN + PCA + segment feedback |
| 10 | Nested CV + calibration + imbalance metrics + significance tests |
| 11 | Versioned champion + shared features + PSI baseline |
| 12 | FastAPI + Docker + health/metrics/drift/retraining |

## Data policy

The full City CSV and generated full station-hour table are intentionally not committed to Git because they are reproducibly downloadable/generated. Small samples and validation reports are committed. Binary `.joblib` artifacts are regenerated by Week 11.

## Viva one-liner

> ChargeUp uses the City of Palo Alto's real ChargePoint sessions to forecast next-hour station demand and next-hour server-termination risk, clusters stations by usage pattern, and deploys the versioned models through FastAPI with drift monitoring.
