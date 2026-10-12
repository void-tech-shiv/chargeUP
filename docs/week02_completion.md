# ChargeUp — Week 2 Completion

Week 2 follows the SLM data-preparation stage and is now complete.

## Completed work

- [x] Reproducible data ingestion
- [x] Required-column/schema validation
- [x] Duplicate-row and duplicate station/timestamp checks
- [x] Timestamp parsing and chronological ordering
- [x] Missing-value handling without imputing target labels
- [x] Domain-validity checks for weather, geography and charging telemetry
- [x] Outlier diagnostics documented with IQR statistics
- [x] Leakage guardrails for next-hour demand forecasting
- [x] Cleaned data generated under `data/processed/`
- [x] Data-quality report generated under `reports/`

## Verification summary

- Raw rows: 5,360
- Clean rows: 5,360
- Stations: 16
- Fault rate: 3.94%
- Missing values after cleaning: 0
- Duplicate rows removed: 0
- Chronological ordering: PASS
- Leakage checks: PASS

## Reproducibility

Run:

```bash
python -m src.week02_cleaning
```

This generates the full cleaned dataset at:

`data/processed/chargeup_sessions_clean.csv`

and the quality report at:

`reports/week02_data_quality.json`

The repository keeps a lightweight cleaned sample for review while the full cleaned dataset is generated reproducibly by the Week 2 script.

## Week 2 status

**COMPLETE**

Next SLM stage: Week 3 — OLS Linear Regression, Gradient Descent and the first MLflow run.
