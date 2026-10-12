# ChargeUp — Week 3: OLS, Gradient Descent, and First MLflow Run

This week follows the SLM Week 3 milestone for the next-hour EV charging-demand regression task.

## Week 3 objectives

1. Build an Ordinary Least Squares (OLS) linear-regression baseline for `next_hour_demand_kwh`.
2. Implement linear regression with Gradient Descent instead of relying only on a library estimator.
3. Use leakage-safe, time-aware train/validation splitting.
4. Standardize numeric inputs using statistics learned only from the training split for Gradient Descent stability.
5. Compare OLS and Gradient Descent predictions on the same validation window.
6. Report regression metrics such as MAE, RMSE, and R².
7. Record the first experiment in MLflow with parameters, metrics, and artifacts.

## Regression target

`next_hour_demand_kwh`

The target represents the charging demand for the next hour at the same station.

## Leakage-safe rule

For an observation at time `t`, only information available at or before `t` may be used as input. `next_hour_demand_kwh` is the label and must never appear in the feature matrix.

The split must preserve chronology. Random shuffling is not allowed for the Week 3 baseline because future observations must not leak into training.

## Planned Week 3 features

The initial linear baseline will use numeric operational, weather, geography, and time-derived features that are available at prediction time, including:

- `temperature_c`
- `humidity_pct`
- `rainfall_mm`
- `latitude`
- `longitude`
- `session_count`
- `avg_session_minutes`
- `avg_power_kw`
- `connector_errors`
- `voltage_v`
- `current_a`
- `availability_pct`
- `energy_kwh`
- hour-of-day derived from `timestamp`
- day-of-week derived from `timestamp`

`station_fault_flag` will not be used as a demand feature in the Week 3 baseline because the demand model should not depend on a future or externally predicted fault label.

## OLS baseline

The OLS model will be trained with a standard linear-regression implementation and evaluated on the held-out chronological validation set.

Required outputs:

- coefficients/intercept
- validation MAE
- validation RMSE
- validation R²

## Gradient Descent baseline

A separate implementation will optimize mean-squared error iteratively.

Required elements:

- weight initialization
- bias term
- learning rate
- number of iterations / convergence condition
- MSE loss calculation
- gradient calculation
- parameter updates
- loss history

The Gradient Descent implementation will use training-only feature scaling and will be compared against OLS on the same split.

## First MLflow run

The Week 3 experiment will log:

- model type
- train/validation row counts
- feature count
- learning rate and iteration count for Gradient Descent
- MAE
- RMSE
- R²
- loss history artifact for Gradient Descent
- Week 3 metrics JSON artifact

Suggested experiment name:

`chargeup-week03-linear-regression`

## Planned files

- `src/week03_linear_regression.py`
- `reports/week03_regression_metrics.json`
- `reports/week03_gradient_descent_loss.csv`
- `docs/week03_completion.md`

MLflow local run data will remain under `mlruns/` and is intentionally excluded from Git because experiment tracking directories are generated artifacts.

## Week 3 completion criteria

- [x] Week 3 plan documented
- [ ] Chronological regression split implemented
- [ ] OLS baseline implemented
- [ ] Gradient Descent implemented from scratch
- [ ] MAE / RMSE / R² reported
- [ ] OLS vs Gradient Descent compared
- [ ] First MLflow run implemented
- [ ] Week 3 outputs generated
- [ ] Week 3 closed
