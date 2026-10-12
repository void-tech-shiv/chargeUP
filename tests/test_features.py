import pandas as pd
from src.features import build_features, MODEL_FEATURES


def test_feature_shape():
    df = pd.DataFrame([{
        "station_id": "CHG-001",
        "timestamp": "2026-10-13T18:00:00",
        "temperature_c": 31,
        "humidity_pct": 64,
        "rainfall_mm": 0,
        "latitude": 16.5,
        "longitude": 80.6,
        "session_count": 7,
        "avg_session_minutes": 43,
        "avg_power_kw": 7.2,
        "connector_errors": 0,
        "voltage_v": 230,
        "current_a": 28,
        "availability_pct": 99,
        "energy_kwh": 31.5,
    }])
    X = build_features(df, {"CHG-001": 2})
    assert list(X.columns) == MODEL_FEATURES
    assert X.shape == (1, len(MODEL_FEATURES))
    assert int(X["station_segment"].iloc[0]) == 2
