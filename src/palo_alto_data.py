"""Download and prepare the real City of Palo Alto ChargePoint dataset.

Source requested by the 25SC2107E ChargeUp project:
https://data.paloalto.gov/datasets/194693/electric-vehicle-charging-station-usage-july-2011-dec-2020/

The historical City portal currently exposes a stable dataset landing page but
its old cityofpaloalto.org URL can return 404. This module first tries the
current City download endpoint, then a public research mirror containing the
same `ChargePoint Data CY20Q4.csv` file so the project remains reproducible.

The processed table is station-hour data. Both supervised targets refer to the
*next exact hour* at the same station:
- next_hour_demand_kwh: energy delivered in that next hour (0 when no session)
- station_fault_flag: 1 when that next hour contains an `Ended By` value with
  "Server" (a transparent weak-label / server-termination proxy)

No weather, voltage, current, or hardware-availability values are fabricated;
they are not present in the provided City dataset.
"""
from __future__ import annotations

import argparse
import json
import shutil
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .palo_alto_schema import (
    SOURCE_REQUIRED_COLUMNS,
    FAULT_PROXY_DESCRIPTION,
)

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "ChargePoint Data CY20Q4.csv"
PROCESSED_PATH = ROOT / "data" / "processed" / "chargeup_sessions_clean.csv"
REPORT_PATH = ROOT / "reports" / "palo_alto_data_quality.json"
SAMPLE_PATH = ROOT / "data" / "processed" / "palo_alto_hourly_sample.csv"

OFFICIAL_PAGE = (
    "https://data.paloalto.gov/datasets/194693/"
    "electric-vehicle-charging-station-usage-july-2011-dec-2020/"
)
OFFICIAL_DOWNLOAD = (
    "https://data.paloalto.gov/datasets/"
    "194693-electric-vehicle-charging-station-usage-july-2011-dec-2020.download/"
)
FULL_MIRROR = (
    "https://raw.githubusercontent.com/fbohu/"
    "Deep-Spatio-Temporal-Forecasting-of-Electrical-Vehicle-Charging-Demand/"
    "main/data/Palo%20Alto/ChargePoint%20Data%20CY20Q4.csv"
)
EXPORT_MIRROR = (
    "https://raw.githubusercontent.com/raveenakagne/Data-Visualization/main/"
    "ElectricVehicleChargingStationUsageJuly2011Dec2020_2797601698338421488.csv"
)

USECOLS = SOURCE_REQUIRED_COLUMNS + [
    "GHG Savings (kg)",
    "Gasoline Savings (gallons)",
]


def download_file(destination: Path, export_only: bool = False) -> dict[str, Any]:
    """Download the City source, falling back to verified public mirrors."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    urls = [EXPORT_MIRROR] if export_only else [OFFICIAL_DOWNLOAD, FULL_MIRROR, EXPORT_MIRROR]
    errors: list[str] = []

    for url in urls:
        tmp = destination.with_suffix(destination.suffix + ".part")
        try:
            request = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0 ChargeUp-ML-Project/2.0",
                    "Accept": "text/csv,*/*",
                },
            )
            with urllib.request.urlopen(request, timeout=120) as response, tmp.open("wb") as fh:
                shutil.copyfileobj(response, fh)
            if tmp.stat().st_size < 1_000:
                raise RuntimeError("downloaded file is unexpectedly small")
            tmp.replace(destination)
            return {
                "downloaded": True,
                "url_used": url,
                "bytes": int(destination.stat().st_size),
                "fallback_used": url != OFFICIAL_DOWNLOAD,
                "errors_before_success": errors,
            }
        except Exception as exc:  # network behavior is environment-dependent
            errors.append(f"{url}: {type(exc).__name__}: {exc}")
            if tmp.exists():
                tmp.unlink()

    raise RuntimeError("Unable to download Palo Alto dataset.\n" + "\n".join(errors))


def _duration_minutes(series: pd.Series) -> pd.Series:
    # ChargePoint uses H:MM:SS and sometimes durations beyond 24 h; pandas
    # Timedelta handles those directly. Malformed values become NaN.
    return pd.to_timedelta(series.astype("string"), errors="coerce").dt.total_seconds() / 60.0


def validate_source_columns(df: pd.DataFrame) -> None:
    missing = [c for c in SOURCE_REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Palo Alto source is missing required columns: {missing}")


def load_source(path: Path) -> pd.DataFrame:
    # Read the header first because public exports can contain a subset of optional columns.
    header = pd.read_csv(path, nrows=0, encoding="utf-8-sig")
    validate_source_columns(header)
    cols = [c for c in USECOLS if c in header.columns]
    return pd.read_csv(path, usecols=cols, encoding="utf-8-sig", low_memory=False)


def clean_sessions(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    df = raw.copy()
    rows_input = int(len(df))

    df["station_id"] = df["Station Name"].astype("string").str.strip()
    df["start_ts"] = pd.to_datetime(df["Start Date"], errors="coerce")
    df["end_ts"] = pd.to_datetime(df["End Date"], errors="coerce")
    df["charging_minutes"] = _duration_minutes(df["Charging Time (hh:mm:ss)"])
    df["total_duration_minutes"] = _duration_minutes(df["Total Duration (hh:mm:ss)"])

    numeric_map = {
        "energy_kwh_session": "Energy (kWh)",
        "latitude": "Latitude",
        "longitude": "Longitude",
        "fee_usd_session": "Fee",
    }
    for new, old in numeric_map.items():
        df[new] = pd.to_numeric(df[old], errors="coerce")

    # Source-validity rules only. Statistical outliers are reported, not globally
    # clipped here, so later time-split model selection stays leakage-safe.
    invalid_start = int(df["start_ts"].isna().sum())
    invalid_energy = int((df["energy_kwh_session"] < 0).fillna(False).sum())
    invalid_duration = int((df["charging_minutes"] < 0).fillna(False).sum())

    df = df.dropna(subset=["station_id", "start_ts", "energy_kwh_session", "latitude", "longitude"])
    df = df[df["station_id"].str.len() > 0]
    df = df[df["energy_kwh_session"] >= 0]
    df = df[(df["charging_minutes"].isna()) | (df["charging_minutes"] >= 0)]

    charging_hours = df["charging_minutes"] / 60.0
    df["session_power_kw"] = np.where(
        charging_hours > 0,
        df["energy_kwh_session"] / charging_hours,
        np.nan,
    )
    # Impossible calculated power values are treated as missing, not clipped.
    df.loc[(df["session_power_kw"] < 0) | (df["session_power_kw"] > 500), "session_power_kw"] = np.nan

    port = df["Port Type"].astype("string").str.casefold().fillna("")
    df["is_level1"] = port.str.contains("level 1", regex=False).astype(float)
    df["is_level2"] = port.str.contains("level 2", regex=False).astype(float)
    df["is_dc_fast"] = port.str.contains("dc", regex=False).astype(float)

    ended = df["Ended By"].astype("string").str.casefold().fillna("")
    df["server_termination"] = ended.str.contains("server", regex=False).astype(int)

    df["fee_usd_session"] = df["fee_usd_session"].fillna(0.0).clip(lower=0)
    df["timestamp"] = df["start_ts"].dt.floor("h")

    info = {
        "rows_input": rows_input,
        "rows_valid_sessions": int(len(df)),
        "rows_removed": int(rows_input - len(df)),
        "invalid_start_timestamp_count": invalid_start,
        "negative_energy_count": invalid_energy,
        "negative_charging_duration_count": invalid_duration,
        "server_terminated_sessions": int(df["server_termination"].sum()),
        "station_count_raw": int(df["station_id"].nunique()),
        "time_min": df["start_ts"].min().isoformat() if len(df) else None,
        "time_max": df["start_ts"].max().isoformat() if len(df) else None,
    }
    return df, info


def aggregate_station_hour(sessions: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        sessions.groupby(["station_id", "timestamp"], as_index=False)
        .agg(
            latitude=("latitude", "median"),
            longitude=("longitude", "median"),
            session_count=("station_id", "size"),
            avg_session_minutes=("charging_minutes", "mean"),
            avg_total_duration_minutes=("total_duration_minutes", "mean"),
            avg_power_kw=("session_power_kw", "mean"),
            energy_kwh=("energy_kwh_session", "sum"),
            fee_usd=("fee_usd_session", "sum"),
            level1_share=("is_level1", "mean"),
            level2_share=("is_level2", "mean"),
            dc_fast_share=("is_dc_fast", "mean"),
            current_server_termination_count=("server_termination", "sum"),
        )
        .sort_values(["timestamp", "station_id"])
        .reset_index(drop=True)
    )

    # A missing mean can occur when charging time is absent. Fill only from the
    # same station's historical median, then from the global median.
    for col in ["avg_session_minutes", "avg_total_duration_minutes", "avg_power_kw"]:
        station_med = grouped.groupby("station_id")[col].transform("median")
        grouped[col] = grouped[col].fillna(station_med)
        grouped[col] = grouped[col].fillna(grouped[col].median()).fillna(0.0)

    # Build exact next-hour labels by shifting the *timestamp key*, not by simply
    # shifting to the next observed record. If no session exists next hour, demand
    # and the server-termination proxy are correctly zero.
    future = grouped[
        ["station_id", "timestamp", "energy_kwh", "current_server_termination_count"]
    ].copy()
    future["timestamp"] = future["timestamp"] - pd.Timedelta(hours=1)
    future = future.rename(
        columns={
            "energy_kwh": "next_hour_demand_kwh",
            "current_server_termination_count": "next_hour_server_termination_count",
        }
    )
    grouped = grouped.merge(future, on=["station_id", "timestamp"], how="left")
    grouped["next_hour_demand_kwh"] = grouped["next_hour_demand_kwh"].fillna(0.0).clip(lower=0)
    grouped["next_hour_server_termination_count"] = grouped[
        "next_hour_server_termination_count"
    ].fillna(0).astype(int)
    grouped["station_fault_flag"] = (
        grouped["next_hour_server_termination_count"] > 0
    ).astype(int)

    grouped["session_count"] = grouped["session_count"].astype(int)
    return grouped.sort_values(["timestamp", "station_id"]).reset_index(drop=True)


def make_report(
    raw: pd.DataFrame,
    sessions: pd.DataFrame,
    hourly: pd.DataFrame,
    source_path: Path,
    download_meta: dict[str, Any] | None,
    clean_info: dict[str, Any],
) -> dict[str, Any]:
    source_fault_rate = float(sessions["server_termination"].mean()) if len(sessions) else 0.0
    target_fault_rate = float(hourly["station_fault_flag"].mean()) if len(hourly) else 0.0
    return {
        "dataset": "City of Palo Alto Electric Vehicle Charging Station Usage (July 2011 - Dec 2020)",
        "official_page": OFFICIAL_PAGE,
        "source_file": str(source_path),
        "download": download_meta,
        "raw_rows": int(len(raw)),
        **clean_info,
        "station_hour_rows": int(len(hourly)),
        "station_count": int(hourly["station_id"].nunique()),
        "station_hour_time_min": hourly["timestamp"].min().isoformat() if len(hourly) else None,
        "station_hour_time_max": hourly["timestamp"].max().isoformat() if len(hourly) else None,
        "source_server_termination_rate": source_fault_rate,
        "classification_target_positive_rate": target_fault_rate,
        "regression_target": "next exact-hour station energy (kWh); 0 if no charging session",
        "classification_target": FAULT_PROXY_DESCRIPTION,
        "leakage_guard": (
            "current_server_termination_count and next_hour_server_termination_count are retained "
            "for audit only and are excluded from all model feature lists"
        ),
        "weather_note": (
            "The provided City ChargePoint CSV has no weather columns. Weather is not fabricated. "
            "The real-data model uses charging-session, time, port, fee and geography features."
        ),
        "source_columns": list(raw.columns),
    }


def prepare(
    source_path: Path = RAW_PATH,
    processed_path: Path = PROCESSED_PATH,
    report_path: Path = REPORT_PATH,
    sample_path: Path = SAMPLE_PATH,
    download: bool = False,
    export_only: bool = False,
) -> dict[str, Any]:
    download_meta = None
    if download or not source_path.exists():
        download_meta = download_file(source_path, export_only=export_only)

    raw = load_source(source_path)
    sessions, clean_info = clean_sessions(raw)
    hourly = aggregate_station_hour(sessions)

    processed_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    hourly.to_csv(processed_path, index=False)
    hourly.head(250).to_csv(sample_path, index=False)

    report = make_report(raw, sessions, hourly, source_path, download_meta, clean_info)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare the real Palo Alto EV charging dataset")
    parser.add_argument("--source", type=Path, default=RAW_PATH)
    parser.add_argument("--output", type=Path, default=PROCESSED_PATH)
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    parser.add_argument("--sample", type=Path, default=SAMPLE_PATH)
    parser.add_argument("--download", action="store_true", help="download/re-download source CSV")
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="use the smaller public 10,000-row portal export mirror (CI/debug only)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = prepare(
        source_path=args.source,
        processed_path=args.output,
        report_path=args.report,
        sample_path=args.sample,
        download=args.download,
        export_only=args.export_only,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
