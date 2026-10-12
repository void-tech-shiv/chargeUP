from __future__ import annotations
import numpy as np
import pandas as pd

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

DERIVED_FEATURES = [
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "is_weekend",
    "is_peak_hour",
    "month",
    "station_segment",
]

MODEL_FEATURES = BASE_FEATURES + DERIVED_FEATURES


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    ts = pd.to_datetime(out["timestamp"], errors="coerce")

    hour = ts.dt.hour.fillna(0)
    dow = ts.dt.dayofweek.fillna(0)

    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    out["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    out["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    out["is_weekend"] = (dow >= 5).astype(int)
    out["is_peak_hour"] = hour.isin([7, 8, 9, 17, 18, 19, 20]).astype(int)
    out["month"] = ts.dt.month.fillna(1).astype(int)
    return out


def build_features(
    df: pd.DataFrame,
    station_segments: dict[str, int] | None = None,
) -> pd.DataFrame:
    out = add_time_features(df)

    station_segments = station_segments or {}
    out["station_segment"] = (
        out["station_id"].astype(str).map(station_segments).fillna(-1).astype(int)
    )

    for col in BASE_FEATURES:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    # Training medians should ideally be persisted. For this compact capstone,
    # missing numeric values are filled with the current batch median, then 0.
    out[MODEL_FEATURES] = out[MODEL_FEATURES].replace([np.inf, -np.inf], np.nan)
    out[MODEL_FEATURES] = out[MODEL_FEATURES].fillna(out[MODEL_FEATURES].median(numeric_only=True))
    out[MODEL_FEATURES] = out[MODEL_FEATURES].fillna(0)

    return out[MODEL_FEATURES].astype(float)
