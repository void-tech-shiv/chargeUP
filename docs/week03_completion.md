# Week 3 Completion — OLS, Gradient Descent, MLflow

SLM scope: OLS Linear Regression, a from-scratch Gradient Descent implementation, regression metrics, and the first MLflow experiment hook.

Completed:
- chronological 80/20 split with a one-hour purge at the forecast boundary
- OLS baseline
- batch Gradient Descent implemented from scratch
- MAE, RMSE and R² comparison
- MLflow experiment/run integration in `src/week03_linear_regression.py`
- reproducible metrics and prediction report

Verified metrics on the full generated dataset: OLS RMSE 5.922, Gradient Descent RMSE 5.922, R² about 0.705. Gradient Descent converged in 519 iterations.

MLflow is listed in `requirements.txt`; the repository-preparation runtime did not have the package preinstalled, so the script records that condition instead of fabricating a run ID. Running after `pip install -r requirements.txt` creates the MLflow run automatically.
