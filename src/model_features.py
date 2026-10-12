"""Shared leakage-safe feature transformation for Palo Alto training/inference."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .palo_alto_schema import BASE_FEATURES,TIME_FEATURES,MODEL_FEATURES

def build_model_features(df: pd.DataFrame, station_segments: dict[str,int] | None=None)->pd.DataFrame:
    x=df.copy(); ts=pd.to_datetime(x['timestamp'],errors='coerce'); h=ts.dt.hour.fillna(0); d=ts.dt.dayofweek.fillna(0)
    x['hour_sin']=np.sin(2*np.pi*h/24); x['hour_cos']=np.cos(2*np.pi*h/24); x['dow_sin']=np.sin(2*np.pi*d/7); x['dow_cos']=np.cos(2*np.pi*d/7); x['is_weekend']=(d>=5).astype(int); x['is_peak_hour']=h.isin([7,8,9,17,18,19,20]).astype(int)
    station_segments=station_segments or {}; x['station_segment']=x['station_id'].astype(str).map(station_segments).fillna(-1).astype(float)
    for col in BASE_FEATURES:
        if col not in x.columns: x[col]=0.0
    out=x[MODEL_FEATURES].apply(pd.to_numeric,errors='coerce').replace([np.inf,-np.inf],np.nan)
    return out.fillna(0.0).astype(float)
