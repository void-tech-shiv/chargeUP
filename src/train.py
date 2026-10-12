from __future__ import annotations
import argparse
import json
from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from xgboost import XGBRegressor, XGBClassifier

from .clustering import fit_station_clusters, save_station_clusters
from .features import build_features, MODEL_FEATURES
from .drift import make_baseline_stats

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw" / "chargeup_sessions.csv"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"


def temporal_split(df: pd.DataFrame, frac: float = 0.80):
    ordered = df.sort_values("timestamp").reset_index(drop=True)
    split_idx = int(len(ordered) * frac)
    return ordered.iloc[:split_idx].copy(), ordered.iloc[split_idx:].copy()


def reg_metrics(y, pred):
    return {
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": float(mean_squared_error(y, pred) ** 0.5),
        "r2": float(r2_score(y, pred)),
    }


def clf_metrics(y, prob, pred):
    out = {
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
    }
    try:
        out["roc_auc"] = float(roc_auc_score(y, prob))
        out["pr_auc"] = float(average_precision_score(y, prob))
    except ValueError:
        out["roc_auc"] = None
        out["pr_auc"] = None
    return out


def time_series_cv_reg(model, X, y, n_splits=3):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []
    for tr, va in tscv.split(X):
        m = clone(model)
        m.fit(X.iloc[tr], y.iloc[tr])
        scores.append(mean_absolute_error(y.iloc[va], m.predict(X.iloc[va])))
    return float(np.mean(scores))


def time_series_cv_clf(model, X, y, n_splits=3):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []
    for tr, va in tscv.split(X):
        ytr, yva = y.iloc[tr], y.iloc[va]
        if ytr.nunique() < 2 or yva.nunique() < 2:
            continue
        m = clone(model)
        m.fit(X.iloc[tr], ytr)
        p = m.predict_proba(X.iloc[va])[:, 1]
        scores.append(average_precision_score(yva, p))
    return float(np.mean(scores)) if scores else 0.0


def maybe_log_mlflow(metrics, enabled=False):
    if not enabled:
        return
    try:
        import mlflow
        mlflow.set_experiment("chargeup")
        with mlflow.start_run(run_name="chargeup_training"):
            for group, values in metrics.items():
                if isinstance(values, dict):
                    for k, v in values.items():
                        if isinstance(v, (int, float)) and v is not None:
                            mlflow.log_metric(f"{group}_{k}", v)
    except Exception as exc:
        print(f"MLflow logging skipped: {exc}")


def train(use_mlflow: bool = False):
    MODELS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    if not DATA.exists():
        from .generate_data import generate
        generate()

    df = pd.read_csv(DATA, parse_dates=["timestamp"])
    df = df.dropna(subset=["next_hour_demand_kwh", "station_fault_flag"]).copy()

    # Week 9 style station usage segmentation.
    segment_map, station_report, cluster_artifacts = fit_station_clusters(df)
    save_station_clusters(segment_map, station_report, cluster_artifacts)

    train_df, test_df = temporal_split(df)
    X_train = build_features(train_df, segment_map)
    X_test = build_features(test_df, segment_map)

    y_reg_train = train_df["next_hour_demand_kwh"].astype(float)
    y_reg_test = test_df["next_hour_demand_kwh"].astype(float)

    y_clf_train = train_df["station_fault_flag"].astype(int)
    y_clf_test = test_df["station_fault_flag"].astype(int)

    reg_models = {
        "linear_regression": Pipeline([
            ("scale", StandardScaler()),
            ("model", LinearRegression()),
        ]),
        "ridge": Pipeline([
            ("scale", StandardScaler()),
            ("model", Ridge(alpha=3.0)),
        ]),
        "random_forest": RandomForestRegressor(
            n_estimators=220, max_depth=14, min_samples_leaf=2,
            random_state=42, n_jobs=-1
        ),
        "xgboost": XGBRegressor(
            n_estimators=320, learning_rate=0.04, max_depth=5,
            subsample=0.9, colsample_bytree=0.9,
            objective="reg:squarederror", random_state=42, n_jobs=4
        ),
    }

    reg_results = {}
    fitted_reg = {}
    for name, model in reg_models.items():
        cv_mae = time_series_cv_reg(model, X_train, y_reg_train)
        model.fit(X_train, y_reg_train)
        pred = model.predict(X_test)
        m = reg_metrics(y_reg_test, pred)
        m["cv_mae"] = cv_mae
        reg_results[name] = m
        fitted_reg[name] = model
        print("REG", name, m)

    reg_champion_name = min(reg_results, key=lambda n: reg_results[n]["mae"])
    reg_champion = fitted_reg[reg_champion_name]

    # Rare faults: class weighting / scale_pos_weight.
    pos = max(int(y_clf_train.sum()), 1)
    neg = max(len(y_clf_train) - pos, 1)
    scale_pos_weight = neg / pos

    clf_models = {
        "logistic_regression": Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=2000, class_weight="balanced", random_state=42
            )),
        ]),
        "decision_tree": DecisionTreeClassifier(
            max_depth=5, min_samples_leaf=12,
            class_weight="balanced", random_state=42
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=260, max_depth=12, min_samples_leaf=3,
            class_weight="balanced_subsample", random_state=42, n_jobs=-1
        ),
        "xgboost": XGBClassifier(
            n_estimators=280, learning_rate=0.04, max_depth=4,
            subsample=0.9, colsample_bytree=0.9,
            objective="binary:logistic", eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
            random_state=42, n_jobs=4
        ),
    }

    clf_results = {}
    fitted_clf = {}
    for name, model in clf_models.items():
        cv_pr_auc = time_series_cv_clf(model, X_train, y_clf_train)
        model.fit(X_train, y_clf_train)
        prob = model.predict_proba(X_test)[:, 1]
        pred = (prob >= 0.5).astype(int)
        m = clf_metrics(y_clf_test, prob, pred)
        m["cv_pr_auc"] = cv_pr_auc
        clf_results[name] = m
        fitted_clf[name] = model
        print("CLF", name, m)

    # Champion by PR-AUC, appropriate for rare positives.
    clf_champion_name = max(
        clf_results,
        key=lambda n: clf_results[n].get("pr_auc") or -1
    )
    clf_champion = fitted_clf[clf_champion_name]

    # Calibration wrapper if enough minority examples.
    if y_clf_train.sum() >= 20:
        try:
            calibrated = CalibratedClassifierCV(
                estimator=clone(clf_champion),
                method="sigmoid",
                cv=3,
            )
            calibrated.fit(X_train, y_clf_train)
            cal_prob = calibrated.predict_proba(X_test)[:, 1]
            cal_pred = (cal_prob >= 0.5).astype(int)
            cal_metrics = clf_metrics(y_clf_test, cal_prob, cal_pred)
            if (cal_metrics.get("pr_auc") or 0) >= (clf_results[clf_champion_name].get("pr_auc") or 0) * 0.95:
                clf_champion = calibrated
                clf_champion_name = clf_champion_name + "_calibrated"
                clf_results["calibrated_champion"] = cal_metrics
        except Exception as exc:
            print("Calibration skipped:", exc)

    joblib.dump(reg_champion, MODELS / "regression_model.joblib")
    joblib.dump(clf_champion, MODELS / "fault_model.joblib")

    baseline = make_baseline_stats(X_train)
    (MODELS / "baseline_stats.json").write_text(
        json.dumps(baseline, indent=2), encoding="utf-8"
    )

    metrics = {
        "dataset": {
            "rows": int(len(df)),
            "stations": int(df["station_id"].nunique()),
            "fault_rate": float(df["station_fault_flag"].mean()),
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
        },
        "regression_champion": reg_champion_name,
        "classification_champion": clf_champion_name,
        "regression_models": reg_results,
        "classification_models": clf_results,
    }

    (MODELS / "training_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    meta = {
        "project": "ChargeUp",
        "regression_target": "next_hour_demand_kwh",
        "classification_target": "station_fault_flag",
        "features": MODEL_FEATURES,
        "regression_champion": reg_champion_name,
        "classification_champion": clf_champion_name,
        "version": "1.0.0",
    }
    (MODELS / "model_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )

    # Test predictions report.
    test_report = test_df[
        ["station_id", "timestamp", "next_hour_demand_kwh", "station_fault_flag"]
    ].copy()
    test_report["predicted_demand_kwh"] = reg_champion.predict(X_test)
    fault_prob = clf_champion.predict_proba(X_test)[:, 1]
    test_report["fault_probability"] = fault_prob
    test_report["predicted_fault"] = (fault_prob >= 0.5).astype(int)
    test_report.to_csv(REPORTS / "test_predictions.csv", index=False)

    # SHAP / feature importance export for XGBoost-compatible tree model.
    try:
        import shap
        shap_model = fitted_reg.get("xgboost")
        sample = X_test.sample(min(300, len(X_test)), random_state=42)
        explainer = shap.TreeExplainer(shap_model)
        values = explainer.shap_values(sample)
        importance = np.abs(values).mean(axis=0)
        pd.DataFrame({
            "feature": X_test.columns,
            "mean_abs_shap": importance,
        }).sort_values("mean_abs_shap", ascending=False).to_csv(
            MODELS / "shap_importance.csv", index=False
        )
    except Exception as exc:
        print("SHAP export skipped:", exc)

    maybe_log_mlflow({
        "regression": reg_results[reg_champion_name.replace("_calibrated", "")],
        "classification": clf_results.get(
            clf_champion_name.replace("_calibrated", ""),
            clf_results.get("calibrated_champion", {})
        ),
    }, enabled=use_mlflow)

    print(f"\nRegression champion: {reg_champion_name}")
    print(f"Classification champion: {clf_champion_name}")
    print(f"Artifacts saved in: {MODELS}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlflow", action="store_true", help="Log a training run to MLflow")
    args = parser.parse_args()
    train(use_mlflow=args.mlflow)
