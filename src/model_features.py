"""Shared leakage-safe feature transformation used by training and FastAPI inference."""
from __future__ import annotations
import numpy as np
import pandas as pd
BASE_FEATURES=["temperature_c","humidity_pct","rainfall_mm","latitude","longitude","session_count","avg_session_minutes","avg_power_kw","connector_errors","voltage_v","current_a","availability_pct","energy_kwh"]
TIME_FEATURES=["hour_sin","hour_cos","dow_sin","dow_cos","is_weekend","is_peak_hour"]
MODEL_FEATURES=BASE_FEATURES+TIME_FEATURES+["station_segment"]

def build_model_features(df: pd.DataFrame, station_segments: dict[str,int] | None=None)->pd.DataFrame:
    x=df.copy(); ts=pd.to_datetime(x['timestamp'],errors='coerce'); h=ts.dt.hour.fillna(0); d=ts.dt.dayofweek.fillna(0)
    x['hour_sin']=np.sin(2*np.pi*h/24); x['hour_cos']=np.cos(2*np.pi*h/24); x['dow_sin']=np.sin(2*np.pi*d/7); x['dow_cos']=np.cos(2*np.pi*d/7); x['is_weekend']=(d>=5).astype(int); x['is_peak_hour']=h.isin([7,8,9,17,18,19,20]).astype(int)
    station_segments=station_segments or {}; x['station_segment']=x['station_id'].astype(str).map(station_segments).fillna(-1).astype(float)
    out=x[MODEL_FEATURES].apply(pd.to_numeric,errors='coerce').replace([np.inf,-np.inf],np.nan)
    return out.fillna(0.0).astype(float)
