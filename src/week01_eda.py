"""ChargeUp — SLM Week 1: target definition and initial EDA.

This module intentionally performs *exploration only*. It does not clean data,
fit models, tune hyperparameters, or apply leakage-prevention transforms; those
belong to later weeks of the SLM plan.

Usage:
    python -m src.week01_eda
    python -m src.week01_eda --data data/raw/chargeup_sessions_sample.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REGRESSION_TARGET = "next_hour_demand_kwh"
CLASSIFICATION_TARGET = "station_fault_flag"

FEATURE_GROUPS = {
    "charging_station_usage": [
        "session_count",
        "avg_session_minutes",
        "avg_power_kw",
        "connector_errors",
        "voltage_v",
        "current_a",
        "availability_pct",
        "energy_kwh",
    ],
    "time": ["timestamp"],
    "weather": ["temperature_c", "humidity_pct", "rainfall_mm"],
    "geography": ["station_id", "latitude", "longitude"],
}


def resolve_default_data_path() -> Path:
    """Use the full generated dataset when present, otherwise the repo sample."""
    full = Path("data/raw/chargeup_sessions.csv")
    sample = Path("data/raw/chargeup_sessions_sample.csv")
    if full.exists():
        return full
    if sample.exists():
        return sample
    raise FileNotFoundError(
        "No ChargeUp dataset found. Expected data/raw/chargeup_sessions.csv "
        "or data/raw/chargeup_sessions_sample.csv."
    )


def validate_columns(df: pd.DataFrame) -> None:
    required = {
        REGRESSION_TARGET,
        CLASSIFICATION_TARGET,
        *(column for columns in FEATURE_GROUPS.values() for column in columns),
    }
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Dataset is missing required Week 1 columns: {missing}")


def print_heading(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def run_eda(data_path: Path) -> None:
    df = pd.read_csv(data_path)
    validate_columns(df)

    timestamps = pd.to_datetime(df["timestamp"], errors="coerce")

    print_heading("ChargeUp — SLM Week 1: Problem Definition & Initial EDA")
    print(f"Dataset: {data_path}")
    print(f"Rows: {len(df):,}")
    print(f"Stations: {df['station_id'].nunique():,}")
    print(f"Time range: {timestamps.min()} to {timestamps.max()}")
    print(f"Total missing values: {int(df.isna().sum().sum()):,}")

    print_heading("1. Machine-learning targets")
    print(f"Classification target : {CLASSIFICATION_TARGET} (0=normal, 1=fault)")
    print(f"Regression target     : {REGRESSION_TARGET} (next-hour demand in kWh)")

    print_heading("2. SLM input groups")
    for group, columns in FEATURE_GROUPS.items():
        print(f"{group}: {', '.join(columns)}")

    print_heading("3. Initial EDA — next-hour demand")
    demand = df[REGRESSION_TARGET]
    print(demand.describe().to_string())

    correlation_candidates = [
        "energy_kwh",
        "session_count",
        "temperature_c",
        "rainfall_mm",
        "availability_pct",
        "connector_errors",
        "humidity_pct",
    ]
    correlations = (
        df[correlation_candidates + [REGRESSION_TARGET]]
        .corr(numeric_only=True)[REGRESSION_TARGET]
        .drop(REGRESSION_TARGET)
        .sort_values(ascending=False)
    )
    print("\nExploratory correlations with next-hour demand:")
    print(correlations.to_string())

    hourly = (
        df.assign(hour=timestamps.dt.hour)
        .groupby("hour", dropna=True)[REGRESSION_TARGET]
        .mean()
        .sort_values(ascending=False)
    )
    print("\nTop 5 hours by mean next-hour demand:")
    print(hourly.head(5).to_string())

    print_heading("4. Initial EDA — station fault flag")
    fault_counts = df[CLASSIFICATION_TARGET].value_counts(dropna=False).sort_index()
    print("Class counts:")
    print(fault_counts.to_string())
    fault_rate = df[CLASSIFICATION_TARGET].mean() * 100
    print(f"\nObserved fault rate: {fault_rate:.2f}%")

    fault_comparison = df.groupby(CLASSIFICATION_TARGET)[
        ["connector_errors", "availability_pct"]
    ].mean()
    print("\nMean connector errors and availability by fault class:")
    print(fault_comparison.to_string())

    print_heading("5. Geography check")
    print(
        f"Latitude range : {df['latitude'].min():.6f} to {df['latitude'].max():.6f}"
    )
    print(
        f"Longitude range: {df['longitude'].min():.6f} to {df['longitude'].max():.6f}"
    )
    print("\nObservations per station:")
    print(df.groupby("station_id").size().sort_values(ascending=False).to_string())

    print_heading("6. Week 1 boundary")
    print(
        "This script only defines the two targets and performs initial EDA. "
        "Cleaning, leakage-safe preparation, model fitting, regularisation, "
        "tree/boosting models, clustering, calibration, deployment, drift "
        "monitoring, and retraining are intentionally left for later SLM weeks."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run ChargeUp SLM Week 1 target definition and initial EDA."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="CSV path. Defaults to full dataset if present, otherwise repo sample.",
    )
    args = parser.parse_args()

    data_path = args.data if args.data is not None else resolve_default_data_path()
    run_eda(data_path)


if __name__ == "__main__":
    main()
