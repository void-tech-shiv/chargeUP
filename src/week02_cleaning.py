"""ChargeUp — SLM Week 2 data ingestion and cleaning.

This module implements the Week 2 data-preparation stage:
- reproducible ingestion
- schema validation
- duplicate and timestamp checks
- missing-value handling
- domain-based outlier handling
- chronological ordering
- leakage guardrails
- cleaned dataset + data-quality report outputs

The script intentionally avoids fitting model-specific statistical transforms.
Those transforms must be learned only from training folds in later weeks.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW = ROOT / "data" / "raw" / "chargeup_sessions.csv"
SAMPLE_RAW = ROOT / "data" / "raw" / "chargeup_sessions_sample.csv"
DEFAULT_CLEAN = ROOT / "data" / "processed" / "chargeup_sessions_clean.csv"
DEFAULT_REPORT = ROOT / "reports" / "week02_data_quality.json"

REQUIRED_COLUMNS = [
    "station_id",
    "timestamp",
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
    "station_fault_flag",
    "next_hour_demand_kwh",
]

TARGET_COLUMNS = {"station_fault_flag", "next_hour_demand_kwh"}

WEATHER_COLUMNS = ["temperature_c", "humidity_pct", "rainfall_mm"]
TELEMETRY_COLUMNS = [
    "session_count",
    "avg_session_minutes",
    "avg_power_kw",
    "connector_errors",
    "voltage_v",
    "current_a",
    "availability_pct",
    "energy_kwh",
]

# Physical / domain validity bounds. Values outside these ranges are treated as
# invalid measurements, not merely statistical outliers.
DOMAIN_BOUNDS: dict[str, tuple[float | None, float | None]] = {
    "temperature_c": (-20.0, 60.0),
    "humidity_pct": (0.0, 100.0),
    "rainfall_mm": (0.0, 250.0),
    "latitude": (-90.0, 90.0),
    "longitude": (-180.0, 180.0),
    "session_count": (0.0, None),
    "avg_session_minutes": (0.0, 24.0 * 60.0),
    "avg_power_kw": (0.0, 400.0),
    "connector_errors": (0.0, None),
    "voltage_v": (150.0, 300.0),
    "current_a": (0.0, 1000.0),
    "availability_pct": (0.0, 100.0),
    "energy_kwh": (0.0, None),
    "next_hour_demand_kwh": (0.0, None),
}


def _python_value(value: Any) -> Any:
    """Convert NumPy/Pandas scalar values into JSON-serializable Python types."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if pd.isna(value):
        return None
    return value


def load_data(path: Path) -> tuple[pd.DataFrame, Path, str | None]:
    """Load a CSV reproducibly, with a sample fallback for cloned repos."""
    warning = None
    source = path

    if not source.exists():
        if path == DEFAULT_RAW and SAMPLE_RAW.exists():
            source = SAMPLE_RAW
            warning = (
                "Full raw dataset was not found; using the repository sample. "
                "Run `python -m src.generate_data` before Week 2 evaluation "
                "to create the full synthetic dataset."
            )
        else:
            raise FileNotFoundError(f"Input dataset not found: {path}")

    df = pd.read_csv(source)
    return df, source, warning


def validate_schema(df: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["station_id"] = out["station_id"].astype("string").str.strip()
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")

    numeric_columns = [c for c in REQUIRED_COLUMNS if c not in {"station_id", "timestamp"}]
    for column in numeric_columns:
        out[column] = pd.to_numeric(out[column], errors="coerce")

    return out


def apply_domain_bounds(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Replace physically impossible values with NaN so they can be handled safely."""
    out = df.copy()
    invalid_counts: dict[str, int] = {}

    for column, (lower, upper) in DOMAIN_BOUNDS.items():
        mask = pd.Series(False, index=out.index)
        if lower is not None:
            mask |= out[column] < lower
        if upper is not None:
            mask |= out[column] > upper

        invalid_counts[column] = int(mask.sum())
        out.loc[mask, column] = np.nan

    # Fault labels are strictly binary.
    invalid_fault = ~out["station_fault_flag"].isin([0, 1]) & out["station_fault_flag"].notna()
    invalid_counts["station_fault_flag"] = int(invalid_fault.sum())
    out.loc[invalid_fault, "station_fault_flag"] = np.nan

    return out, invalid_counts


def handle_missing_values(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Handle missing values without inventing target labels.

    Rules:
    - identifiers, timestamps and targets are never imputed; rows missing them are dropped
    - weather fields use station-wise time interpolation, then global median fallback
    - telemetry fields use station median, then global median fallback
    - geography uses station median, then global median fallback
    """
    out = df.copy()
    before = out.isna().sum().to_dict()

    required_non_imputable = [
        "station_id",
        "timestamp",
        "station_fault_flag",
        "next_hour_demand_kwh",
    ]
    out = out.dropna(subset=required_non_imputable)
    out = out[out["station_id"].str.len() > 0]
    out = out.sort_values(["station_id", "timestamp"]).reset_index(drop=True)

    # Time-aware interpolation for weather, performed only within a station.
    for column in WEATHER_COLUMNS:
        out[column] = (
            out.groupby("station_id", group_keys=False)[column]
            .apply(lambda s: s.interpolate(method="linear", limit_direction="both"))
            .reset_index(level=0, drop=True)
        )
        median = out[column].median()
        if pd.notna(median):
            out[column] = out[column].fillna(median)

    for column in ["latitude", "longitude"] + TELEMETRY_COLUMNS:
        station_median = out.groupby("station_id")[column].transform("median")
        out[column] = out[column].fillna(station_median)
        global_median = out[column].median()
        if pd.notna(global_median):
            out[column] = out[column].fillna(global_median)

    # Any row still containing a required field that could not be repaired is removed.
    out = out.dropna(subset=REQUIRED_COLUMNS).reset_index(drop=True)

    after = out.isna().sum().to_dict()
    handled = {
        column: int(before.get(column, 0) - after.get(column, 0))
        for column in REQUIRED_COLUMNS
        if int(before.get(column, 0)) > 0
    }
    return out, handled


def remove_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    out = df.copy()
    exact_duplicates = int(out.duplicated().sum())
    duplicate_station_time = int(out.duplicated(subset=["station_id", "timestamp"]).sum())

    out = out.drop_duplicates()
    # One station should contribute at most one row for a given hourly timestamp.
    out = out.drop_duplicates(subset=["station_id", "timestamp"], keep="first")

    return out.reset_index(drop=True), {
        "exact_duplicates_removed": exact_duplicates,
        "duplicate_station_timestamp_removed": duplicate_station_time,
    }


def leakage_checks(df: pd.DataFrame) -> dict[str, Any]:
    """Check Week 2 leakage guardrails before later model construction."""
    suspicious_future_columns = [
        c
        for c in df.columns
        if c.lower().startswith("future_")
        or c.lower().endswith("_future")
        or "next_hour" in c.lower() and c != "next_hour_demand_kwh"
    ]

    # The next-hour demand label is allowed only as the regression target.
    feature_candidates = [c for c in df.columns if c not in TARGET_COLUMNS]
    target_in_features = sorted(TARGET_COLUMNS.intersection(feature_candidates))

    chronology_ok = True
    for _, group in df.groupby("station_id", sort=False):
        if not group["timestamp"].is_monotonic_increasing:
            chronology_ok = False
            break

    return {
        "target_columns": sorted(TARGET_COLUMNS),
        "target_in_feature_candidates": target_in_features,
        "suspicious_future_columns": suspicious_future_columns,
        "chronological_ordering_ok": chronology_ok,
        "policy": (
            "At time t, later modeling stages may use only values available at or before t. "
            "next_hour_demand_kwh is a label and must never be included as an input feature."
        ),
    }


def iqr_diagnostics(df: pd.DataFrame) -> dict[str, dict[str, float | int]]:
    """Report statistical outliers without clipping them globally.

    Statistical clipping is intentionally deferred because thresholds must be learned
    on training data only to avoid leakage into validation/test folds.
    """
    diagnostics: dict[str, dict[str, float | int]] = {}
    columns = WEATHER_COLUMNS + TELEMETRY_COLUMNS + ["next_hour_demand_kwh"]

    for column in columns:
        series = df[column].dropna()
        if series.empty:
            continue
        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        count = int(((series < lower) | (series > upper)).sum())
        diagnostics[column] = {
            "q1": q1,
            "q3": q3,
            "iqr": iqr,
            "lower_fence": lower,
            "upper_fence": upper,
            "iqr_outlier_count": count,
        }

    return diagnostics


def build_quality_report(
    raw: pd.DataFrame,
    cleaned: pd.DataFrame,
    source: Path,
    warning: str | None,
    invalid_counts: dict[str, int],
    duplicate_info: dict[str, int],
    missing_handled: dict[str, int],
) -> dict[str, Any]:
    return {
        "week": 2,
        "source_file": str(source.relative_to(ROOT) if source.is_relative_to(ROOT) else source),
        "warning": warning,
        "rows_raw": int(len(raw)),
        "rows_clean": int(len(cleaned)),
        "rows_removed": int(len(raw) - len(cleaned)),
        "stations": int(cleaned["station_id"].nunique()),
        "time_min": cleaned["timestamp"].min().isoformat() if not cleaned.empty else None,
        "time_max": cleaned["timestamp"].max().isoformat() if not cleaned.empty else None,
        "fault_rate": float(cleaned["station_fault_flag"].mean()) if not cleaned.empty else None,
        "missing_values_before": {
            k: int(v) for k, v in raw.isna().sum().to_dict().items()
        },
        "missing_values_after": {
            k: int(v) for k, v in cleaned.isna().sum().to_dict().items()
        },
        "missing_values_handled": missing_handled,
        "invalid_domain_values_detected": invalid_counts,
        "duplicates": duplicate_info,
        "iqr_diagnostics": iqr_diagnostics(cleaned),
        "leakage_checks": leakage_checks(cleaned),
        "outlier_policy": (
            "Physically impossible values are converted to missing and handled. "
            "IQR outliers are reported but not globally clipped during Week 2; "
            "statistical thresholds must be fit on training folds in later weeks."
        ),
    }


def clean_dataset(input_path: Path = DEFAULT_RAW) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw, source, warning = load_data(input_path)
    validate_schema(raw)

    working = coerce_types(raw)
    working, duplicate_info = remove_duplicates(working)
    working, invalid_counts = apply_domain_bounds(working)
    cleaned, missing_handled = handle_missing_values(working)

    cleaned["station_fault_flag"] = cleaned["station_fault_flag"].astype(int)
    cleaned["session_count"] = cleaned["session_count"].round().astype(int)
    cleaned["connector_errors"] = cleaned["connector_errors"].round().astype(int)
    cleaned = cleaned.sort_values(["station_id", "timestamp"]).reset_index(drop=True)

    report = build_quality_report(
        raw=raw,
        cleaned=cleaned,
        source=source,
        warning=warning,
        invalid_counts=invalid_counts,
        duplicate_info=duplicate_info,
        missing_handled=missing_handled,
    )

    if not report["leakage_checks"]["chronological_ordering_ok"]:
        raise ValueError("Chronological ordering check failed after cleaning.")
    if report["leakage_checks"]["suspicious_future_columns"]:
        raise ValueError(
            "Potential leakage columns detected: "
            f"{report['leakage_checks']['suspicious_future_columns']}"
        )

    return cleaned, report


def save_outputs(
    cleaned: pd.DataFrame,
    report: dict[str, Any],
    clean_path: Path = DEFAULT_CLEAN,
    report_path: Path = DEFAULT_REPORT,
) -> None:
    clean_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    cleaned.to_csv(clean_path, index=False)
    report_path.write_text(json.dumps(report, indent=2, default=_python_value), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ChargeUp SLM Week 2 data cleaning")
    parser.add_argument("--input", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--output", type=Path, default=DEFAULT_CLEAN)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cleaned, report = clean_dataset(args.input)
    save_outputs(cleaned, report, args.output, args.report)

    print("ChargeUp — Week 2 data preparation")
    if report.get("warning"):
        print(f"WARNING: {report['warning']}")
    print(f"Raw rows:     {report['rows_raw']:,}")
    print(f"Clean rows:   {report['rows_clean']:,}")
    print(f"Stations:     {report['stations']}")
    print(f"Fault rate:   {report['fault_rate']:.2%}")
    print(f"Clean output: {args.output}")
    print(f"Report:       {args.report}")


if __name__ == "__main__":
    main()
