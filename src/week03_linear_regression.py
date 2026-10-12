"""ChargeUp — SLM Week 3: OLS, gradient descent, and first MLflow run.

Week 3 scope only:
- next-hour demand regression target
- chronological leakage-safe train/test split
- Ordinary Least Squares (OLS) baseline
- linear regression trained with gradient descent from scratch
- MAE, RMSE, and R^2 evaluation
- first MLflow experiment/run

Ridge/Lasso and broader feature engineering are intentionally deferred to Week 4.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CLEAN = ROOT / "data" / "processed" / "chargeup_sessions_clean.csv"
CLEAN_SAMPLE = ROOT / "data" / "processed" / "chargeup_sessions_clean_sample.csv"
RAW_FULL = ROOT / "data" / "raw" / "chargeup_sessions.csv"
RAW_SAMPLE = ROOT / "data" / "raw" / "chargeup_sessions_sample.csv"
DEFAULT_METRICS = ROOT / "reports" / "week03_linear_regression_metrics.json"
DEFAULT_PREDICTIONS = ROOT / "reports" / "week03_test_predictions.csv"

TARGET = "next_hour_demand_kwh"

# Basic Week 3 predictors. Only direct measurements plus simple calendar fields are
# used here. Richer engineered features are deliberately postponed to Week 4.
BASE_FEATURES = [
    "temperature_c",
    "humidity_pct",
    "rainfall_mm",
    "latitude",
    "longitude",
    "session_count",
    "avg_session_minutes",
    "avg_power_kw",
    "connector_errors",
    "voltage_v",
    "current_a",
    "availability_pct",
    "energy_kwh",
]
TIME_FEATURES = ["hour", "day_of_week"]
FEATURES = BASE_FEATURES + TIME_FEATURES


class GradientDescentLinearRegressor:
    """Linear regression optimized by batch gradient descent from scratch."""

    def __init__(
        self,
        learning_rate: float = 0.03,
        max_iter: int = 5000,
        tol: float = 1e-10,
    ) -> None:
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.x_mean_: np.ndarray | None = None
        self.x_scale_: np.ndarray | None = None
        self.y_mean_: float | None = None
        self.y_scale_: float | None = None
        self.coef_: np.ndarray | None = None
        self.intercept_: float | None = None
        self.n_iter_: int = 0
        self.final_loss_: float | None = None
        self.loss_history_: list[dict[str, float | int]] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GradientDescentLinearRegressor":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)

        if X.ndim != 2 or len(X) != len(y):
            raise ValueError("X must be 2D and have the same number of rows as y.")
        if len(X) == 0:
            raise ValueError("Cannot fit gradient descent on an empty dataset.")

        self.x_mean_ = X.mean(axis=0)
        self.x_scale_ = X.std(axis=0)
        self.x_scale_[self.x_scale_ == 0] = 1.0

        self.y_mean_ = float(y.mean())
        y_scale = float(y.std())
        self.y_scale_ = y_scale if y_scale > 0 else 1.0

        Xs = (X - self.x_mean_) / self.x_scale_
        ys = (y - self.y_mean_) / self.y_scale_

        weights = np.zeros(Xs.shape[1], dtype=float)
        bias = 0.0
        previous_loss = np.inf
        n = len(Xs)

        for iteration in range(1, self.max_iter + 1):
            prediction = Xs @ weights + bias
            error = prediction - ys
            loss = float(np.mean(error ** 2))

            grad_w = (2.0 / n) * (Xs.T @ error)
            grad_b = float((2.0 / n) * error.sum())

            weights -= self.learning_rate * grad_w
            bias -= self.learning_rate * grad_b

            if iteration == 1 or iteration % 100 == 0:
                self.loss_history_.append({"iteration": iteration, "mse_scaled": loss})

            if abs(previous_loss - loss) < self.tol:
                self.n_iter_ = iteration
                previous_loss = loss
                break

            previous_loss = loss
        else:
            self.n_iter_ = self.max_iter

        self.coef_ = weights
        self.intercept_ = bias
        self.final_loss_ = float(previous_loss)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if any(
            value is None
            for value in (
                self.x_mean_,
                self.x_scale_,
                self.y_mean_,
                self.y_scale_,
                self.coef_,
                self.intercept_,
            )
        ):
            raise RuntimeError("Model must be fitted before predict().")

        X = np.asarray(X, dtype=float)
        Xs = (X - self.x_mean_) / self.x_scale_
        scaled_prediction = Xs @ self.coef_ + self.intercept_
        return scaled_prediction * self.y_scale_ + self.y_mean_


def resolve_input(path: Path) -> tuple[Path, str | None]:
    if path.exists():
        return path, None

    if path == DEFAULT_CLEAN:
        for candidate, message in [
            (
                CLEAN_SAMPLE,
                "Full cleaned dataset not found; using cleaned sample. Run Week 2 first for full evaluation.",
            ),
            (
                RAW_FULL,
                "Week 2 cleaned file not found; using full raw dataset. Run Week 2 before final submission.",
            ),
            (
                RAW_SAMPLE,
                "Only raw sample is available; results are demonstration-only.",
            ),
        ]:
            if candidate.exists():
                return candidate, message

    raise FileNotFoundError(f"Dataset not found: {path}")


def load_week3_frame(path: Path) -> tuple[pd.DataFrame, Path, str | None]:
    source, warning = resolve_input(path)
    df = pd.read_csv(source)

    required = {"station_id", "timestamp", TARGET, *BASE_FEATURES}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Week 3 dataset is missing columns: {missing}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["station_id", "timestamp", TARGET]).copy()

    numeric_columns = BASE_FEATURES + [TARGET]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # Basic timestamp representation only; richer feature engineering belongs to Week 4.
    df["hour"] = df["timestamp"].dt.hour.astype(float)
    df["day_of_week"] = df["timestamp"].dt.dayofweek.astype(float)

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=FEATURES + [TARGET])
    df = df.sort_values(["timestamp", "station_id"]).reset_index(drop=True)

    if len(df) < 20:
        raise ValueError("Week 3 needs at least 20 valid rows for chronological evaluation.")

    return df, source, warning


def chronological_split(
    df: pd.DataFrame,
    test_fraction: float = 0.20,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    if not 0 < test_fraction < 0.5:
        raise ValueError("test_fraction must be between 0 and 0.5.")

    unique_times = np.array(sorted(df["timestamp"].dropna().unique()))
    if len(unique_times) < 5:
        raise ValueError("Not enough unique timestamps for a chronological split.")

    split_index = int(np.floor(len(unique_times) * (1.0 - test_fraction)))
    split_index = min(max(split_index, 2), len(unique_times) - 1)
    split_time = pd.Timestamp(unique_times[split_index])

    # Purge the immediately preceding hour because its next-hour target can overlap
    # the first test timestamp. This makes the forecasting boundary explicit.
    purge_boundary = split_time - pd.Timedelta(hours=1)
    train = df[df["timestamp"] < purge_boundary].copy()
    test = df[df["timestamp"] >= split_time].copy()

    if train.empty or test.empty:
        raise ValueError("Chronological split produced an empty train or test set.")
    if train["timestamp"].max() >= test["timestamp"].min():
        raise ValueError("Chronological leakage check failed.")

    return train, test, split_time


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def train_and_evaluate(
    df: pd.DataFrame,
    test_fraction: float,
    learning_rate: float,
    max_iter: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    train, test, split_time = chronological_split(df, test_fraction=test_fraction)

    X_train = train[FEATURES].to_numpy(dtype=float)
    y_train = train[TARGET].to_numpy(dtype=float)
    X_test = test[FEATURES].to_numpy(dtype=float)
    y_test = test[TARGET].to_numpy(dtype=float)

    ols = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("regressor", LinearRegression()),
        ]
    )
    ols.fit(X_train, y_train)
    ols_pred = ols.predict(X_test)

    gd = GradientDescentLinearRegressor(
        learning_rate=learning_rate,
        max_iter=max_iter,
    )
    gd.fit(X_train, y_train)
    gd_pred = gd.predict(X_test)

    ols_metrics = regression_metrics(y_test, ols_pred)
    gd_metrics = regression_metrics(y_test, gd_pred)

    results: dict[str, Any] = {
        "week": 3,
        "target": TARGET,
        "features": FEATURES,
        "split": {
            "strategy": "chronological 80/20 with one-hour purge before test boundary",
            "split_time": split_time.isoformat(),
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "train_time_min": train["timestamp"].min().isoformat(),
            "train_time_max": train["timestamp"].max().isoformat(),
            "test_time_min": test["timestamp"].min().isoformat(),
            "test_time_max": test["timestamp"].max().isoformat(),
        },
        "ols": ols_metrics,
        "gradient_descent": {
            **gd_metrics,
            "learning_rate": learning_rate,
            "max_iter": max_iter,
            "iterations_run": gd.n_iter_,
            "final_scaled_mse": gd.final_loss_,
            "loss_history": gd.loss_history_,
        },
        "comparison": {
            "rmse_difference_gd_minus_ols": gd_metrics["rmse"] - ols_metrics["rmse"],
            "mae_difference_gd_minus_ols": gd_metrics["mae"] - ols_metrics["mae"],
        },
    }

    predictions = test[["station_id", "timestamp", TARGET]].copy()
    predictions["ols_prediction_kwh"] = ols_pred
    predictions["gradient_descent_prediction_kwh"] = gd_pred
    return results, predictions


def log_first_mlflow_run(results: dict[str, Any], metrics_path: Path, predictions_path: Path) -> dict[str, Any]:
    try:
        import mlflow
    except Exception as exc:  # pragma: no cover - environment-dependent
        return {"logged": False, "reason": f"MLflow import failed: {exc}"}

    try:
        mlflow.set_experiment("ChargeUp-Week03-Linear-Regression")
        with mlflow.start_run(run_name="week03-ols-vs-gradient-descent") as run:
            mlflow.log_param("target", TARGET)
            mlflow.log_param("feature_count", len(FEATURES))
            mlflow.log_param("split_strategy", "chronological_with_1h_purge")
            mlflow.log_param(
                "gd_learning_rate",
                results["gradient_descent"]["learning_rate"],
            )
            mlflow.log_param(
                "gd_max_iter",
                results["gradient_descent"]["max_iter"],
            )

            for model_name in ("ols", "gradient_descent"):
                for metric in ("mae", "rmse", "r2"):
                    mlflow.log_metric(
                        f"{model_name}_{metric}",
                        float(results[model_name][metric]),
                    )

            mlflow.log_artifact(str(metrics_path))
            mlflow.log_artifact(str(predictions_path))
            return {"logged": True, "run_id": run.info.run_id}
    except Exception as exc:  # pragma: no cover - environment-dependent
        return {"logged": False, "reason": f"MLflow logging failed: {exc}"}


def save_outputs(
    results: dict[str, Any],
    predictions: pd.DataFrame,
    metrics_path: Path,
    predictions_path: Path,
) -> None:
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    predictions.to_csv(predictions_path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ChargeUp SLM Week 3 linear regression")
    parser.add_argument("--input", type=Path, default=DEFAULT_CLEAN)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--test-fraction", type=float, default=0.20)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--max-iter", type=int, default=5000)
    parser.add_argument("--no-mlflow", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df, source, warning = load_week3_frame(args.input)
    results, predictions = train_and_evaluate(
        df,
        test_fraction=args.test_fraction,
        learning_rate=args.learning_rate,
        max_iter=args.max_iter,
    )
    results["source_file"] = str(source.relative_to(ROOT) if source.is_relative_to(ROOT) else source)
    results["warning"] = warning

    # Save once before MLflow so the first run can log the report artifacts.
    save_outputs(results, predictions, args.metrics, args.predictions)

    if args.no_mlflow:
        results["mlflow"] = {"logged": False, "reason": "disabled by --no-mlflow"}
    else:
        results["mlflow"] = log_first_mlflow_run(results, args.metrics, args.predictions)

    # Save again so the report also records the MLflow run ID/status.
    save_outputs(results, predictions, args.metrics, args.predictions)

    print("ChargeUp — SLM Week 3")
    if warning:
        print(f"WARNING: {warning}")
    print(f"Source: {results['source_file']}")
    print(f"Train rows: {results['split']['train_rows']:,}")
    print(f"Test rows:  {results['split']['test_rows']:,}")
    print(
        "OLS: "
        f"MAE={results['ols']['mae']:.3f}, "
        f"RMSE={results['ols']['rmse']:.3f}, "
        f"R2={results['ols']['r2']:.3f}"
    )
    print(
        "Gradient Descent: "
        f"MAE={results['gradient_descent']['mae']:.3f}, "
        f"RMSE={results['gradient_descent']['rmse']:.3f}, "
        f"R2={results['gradient_descent']['r2']:.3f}, "
        f"iterations={results['gradient_descent']['iterations_run']}"
    )
    print(f"MLflow: {results['mlflow']}")
    print(f"Metrics: {args.metrics}")
    print(f"Predictions: {args.predictions}")


if __name__ == "__main__":
    main()
