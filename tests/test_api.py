from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict():
    payload = {
        "station_id": "CHG-001",
        "timestamp": "2026-10-13T18:00:00",
        "temperature_c": 31,
        "humidity_pct": 64,
        "rainfall_mm": 0,
        "latitude": 16.5062,
        "longitude": 80.6480,
        "session_count": 7,
        "avg_session_minutes": 43,
        "avg_power_kw": 7.2,
        "connector_errors": 0,
        "voltage_v": 230,
        "current_a": 28,
        "availability_pct": 99,
        "energy_kwh": 31.5
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert "next_hour_demand_kwh" in body
    assert "fault_probability" in body
