from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.features import build_features
from src.drift import check_drift

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
DATA = ROOT / "data" / "raw" / "chargeup_sessions.csv"
STATIC = ROOT / "api" / "static"

app = FastAPI(
    title="ChargeUp API",
    version="1.0.0",
    description="Next-hour EV charging demand forecasting + station fault alerting.",
)

app.mount("/static", StaticFiles(directory=STATIC), name="static")

_runtime = {
    "prediction_requests": 0,
    "drift_checks": 0,
    "retrain_requests": 0,
    "started_at": datetime.now(timezone.utc).isoformat(),
}

reg_model = None
fault_model = None
station_segments = {}
baseline_stats = {}
training_metrics = {}
model_meta = {}


def load_artifacts():
    global reg_model, fault_model, station_segments, baseline_stats, training_metrics, model_meta

    required = [
        MODELS / "regression_model.joblib",
        MODELS / "fault_model.joblib",
        MODELS / "station_segments.json",
        MODELS / "baseline_stats.json",
        MODELS / "training_metrics.json",
        MODELS / "model_meta.json",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise RuntimeError(
            "Missing trained artifacts. Run: python -m src.generate_data && python -m src.train\n"
            + "\n".join(missing)
        )

    reg_model = joblib.load(MODELS / "regression_model.joblib")
    fault_model = joblib.load(MODELS / "fault_model.joblib")
    station_segments = json.loads((MODELS / "station_segments.json").read_text())
    baseline_stats = json.loads((MODELS / "baseline_stats.json").read_text())
    training_metrics = json.loads((MODELS / "training_metrics.json").read_text())
    model_meta = json.loads((MODELS / "model_meta.json").read_text())


load_artifacts()


class ChargeUpInput(BaseModel):
    station_id: str = "CHG-001"
    timestamp: datetime = Field(default_factory=datetime.now)
    temperature_c: float = 31.0
    humidity_pct: float = 64.0
    rainfall_mm: float = 0.0
    latitude: float = 16.5062
    longitude: float = 80.6480
    session_count: int = 7
    avg_session_minutes: float = 43.0
    avg_power_kw: float = 7.2
    connector_errors: int = 0
    voltage_v: float = 230.0
    current_a: float = 28.0
    availability_pct: float = 99.0
    energy_kwh: float = 31.5


def to_frame(item: ChargeUpInput) -> pd.DataFrame:
    return pd.DataFrame([item.model_dump()])


def predict_one(item: ChargeUpInput) -> dict:
    df = to_frame(item)
    X = build_features(df, station_segments)

    demand = max(0.0, float(reg_model.predict(X)[0]))
    fault_prob = float(fault_model.predict_proba(X)[0, 1])
    fault = int(fault_prob >= 0.5)

    _runtime["prediction_requests"] += 1

    return {
        "station_id": item.station_id,
        "next_hour_demand_kwh": round(demand, 3),
        "fault_probability": round(fault_prob, 4),
        "station_fault_flag": fault,
        "fault_label": "FAULT" if fault else "NORMAL",
        "station_segment": int(station_segments.get(item.station_id, -1)),
        "model_versions": {
            "regression": model_meta.get("regression_champion"),
            "classification": model_meta.get("classification_champion"),
            "version": model_meta.get("version"),
        },
    }


@app.get("/")
def dashboard():
    return FileResponse(STATIC / "index.html")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "ChargeUp",
        "version": model_meta.get("version"),
        "models_loaded": bool(reg_model is not None and fault_model is not None),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/metrics")
def metrics():
    return {
        "runtime": _runtime,
        "training": training_metrics,
        "model_meta": model_meta,
    }


@app.post("/predict")
def predict(item: ChargeUpInput):
    return predict_one(item)


@app.post("/predict/demand")
def predict_demand(item: ChargeUpInput):
    result = predict_one(item)
    return {
        "station_id": result["station_id"],
        "next_hour_demand_kwh": result["next_hour_demand_kwh"],
        "station_segment": result["station_segment"],
    }


@app.post("/predict/fault")
def predict_fault(item: ChargeUpInput):
    result = predict_one(item)
    return {
        "station_id": result["station_id"],
        "fault_probability": result["fault_probability"],
        "station_fault_flag": result["station_fault_flag"],
        "fault_label": result["fault_label"],
    }


@app.post("/drift/check")
def drift_check(items: List[ChargeUpInput]):
    if not items:
        raise HTTPException(status_code=400, detail="Provide at least one observation.")
    df = pd.DataFrame([i.model_dump() for i in items])
    X = build_features(df, station_segments)
    _runtime["drift_checks"] += 1
    return check_drift(X, baseline_stats)


@app.post("/retrain/trigger")
def retrain_trigger():
    flag = MODELS / "retrain_requested.flag"
    flag.write_text(datetime.now(timezone.utc).isoformat(), encoding="utf-8")
    _runtime["retrain_requests"] += 1
    return {
        "status": "queued",
        "message": "Retraining request recorded.",
        "next_step": "Run `python -m src.train` (or connect this endpoint to a production job queue).",
    }


@app.get("/demo/stations")
def demo_stations(limit: int = 8):
    if not DATA.exists():
        return []

    df = pd.read_csv(DATA, parse_dates=["timestamp"])
    latest = (
        df.sort_values("timestamp")
        .groupby("station_id", as_index=False)
        .tail(1)
        .head(max(1, min(limit, 20)))
    )

    out = []
    for _, row in latest.iterrows():
        item = ChargeUpInput(
            station_id=str(row["station_id"]),
            timestamp=pd.Timestamp(row["timestamp"]).to_pydatetime(),
            temperature_c=float(row["temperature_c"]),
            humidity_pct=float(row["humidity_pct"]),
            rainfall_mm=float(row["rainfall_mm"]),
            latitude=float(row["latitude"]),
            longitude=float(row["longitude"]),
            session_count=int(row["session_count"]),
            avg_session_minutes=float(row["avg_session_minutes"]),
            avg_power_kw=float(row["avg_power_kw"]),
            connector_errors=int(row["connector_errors"]),
            voltage_v=float(row["voltage_v"]),
            current_a=float(row["current_a"]),
            availability_pct=float(row["availability_pct"]),
            energy_kwh=float(row["energy_kwh"]),
        )
        out.append(predict_one(item))
    return out
