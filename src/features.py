"""Compatibility wrapper around the canonical Palo Alto feature builder."""
from __future__ import annotations
import pandas as pd
from .model_features import build_model_features,MODEL_FEATURES
from .palo_alto_schema import BASE_FEATURES

DERIVED_FEATURES=[c for c in MODEL_FEATURES if c not in BASE_FEATURES]

def add_time_features(df:pd.DataFrame)->pd.DataFrame:
    # Retained for backward compatibility; canonical inference uses build_features.
    x=build_model_features(df,{}).copy()
    return x

def build_features(df:pd.DataFrame,station_segments:dict[str,int]|None=None)->pd.DataFrame:
    return build_model_features(df,station_segments)
