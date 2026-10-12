from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "chargeup_sessions.csv"


def generate(seed: int = 42, n_stations: int = 16, days: int = 14) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    station_ids = [f"CHG-{i:03d}" for i in range(1, n_stations + 1)]
    # Vijayawada-ish coordinate center; geography is simply a demo feature.
    center_lat, center_lon = 16.5062, 80.6480

    station_meta = {}
    for i, station_id in enumerate(station_ids):
        station_meta[station_id] = {
            "latitude": center_lat + rng.normal(0, 0.07),
            "longitude": center_lon + rng.normal(0, 0.07),
            "base_demand": rng.uniform(12, 32),
            "power_rating": rng.choice([7.2, 11.0, 22.0, 50.0]),
            "health_bias": rng.uniform(0.0, 1.0),
        }

    start = pd.Timestamp("2026-09-01 00:00:00")
    rows = []

    for station_id in station_ids:
        meta = station_meta[station_id]

        for h in range(days * 24):
            ts = start + pd.Timedelta(hours=h)
            hour = ts.hour
            dow = ts.dayofweek

            morning = np.exp(-((hour - 9) / 3.0) ** 2)
            evening = np.exp(-((hour - 18) / 3.2) ** 2)
            weekend_factor = 0.88 if dow >= 5 else 1.0
            trend = 1.0 + 0.0018 * h

            temperature = 28 + 5 * np.sin(2 * np.pi * (hour - 10) / 24) + rng.normal(0, 1.3)
            humidity = np.clip(67 - 0.8 * (temperature - 28) + rng.normal(0, 5), 35, 95)
            rainfall = max(0.0, rng.gamma(0.7, 1.5) - 0.7) if rng.random() < 0.14 else 0.0

            weather_factor = 1.0 - min(rainfall, 8) * 0.018
            demand = (
                meta["base_demand"]
                * (0.45 + 0.58 * morning + 0.78 * evening)
                * weekend_factor
                * trend
                * weather_factor
                + rng.normal(0, 2.2)
            )
            demand = max(1.0, demand)

            session_count = max(1, int(rng.poisson(max(demand / 4.5, 1))))
            avg_session_minutes = np.clip(rng.normal(42 + demand * 0.25, 8), 15, 110)
            avg_power_kw = max(
                3.0,
                min(meta["power_rating"], rng.normal(meta["power_rating"] * 0.72, 2.0))
            )

            # Rare faults driven by telemetry abnormalities plus random incidents.
            connector_errors = int(rng.poisson(0.10 + meta["health_bias"] * 0.15))
            voltage = rng.normal(230, 3.0)
            current = max(2.0, rng.normal(min(32, avg_power_kw * 1000 / 230), 2.5))
            availability = np.clip(rng.normal(99.1, 1.0), 90, 100)

            anomaly = 0.0
            if rng.random() < 0.028:
                connector_errors += int(rng.integers(2, 7))
                voltage += rng.choice([-1, 1]) * rng.uniform(12, 28)
                availability -= rng.uniform(10, 28)
                anomaly += 2.5

            fault_logit = (
                -5.0
                + 0.72 * connector_errors
                + 0.09 * abs(voltage - 230)
                + 0.10 * max(0, 96 - availability)
                + anomaly
            )
            fault_prob = 1 / (1 + np.exp(-fault_logit))
            fault = int(rng.random() < fault_prob)

            delivered_energy = demand * (0.50 if fault else 1.0) + rng.normal(0, 1.0)
            delivered_energy = max(0.2, delivered_energy)

            rows.append({
                "station_id": station_id,
                "timestamp": ts,
                "temperature_c": round(float(temperature), 3),
                "humidity_pct": round(float(humidity), 3),
                "rainfall_mm": round(float(rainfall), 3),
                "latitude": round(float(meta["latitude"]), 6),
                "longitude": round(float(meta["longitude"]), 6),
                "session_count": session_count,
                "avg_session_minutes": round(float(avg_session_minutes), 3),
                "avg_power_kw": round(float(avg_power_kw), 3),
                "connector_errors": connector_errors,
                "voltage_v": round(float(voltage), 3),
                "current_a": round(float(current), 3),
                "availability_pct": round(float(max(0, availability)), 3),
                "energy_kwh": round(float(delivered_energy), 3),
                "station_fault_flag": fault,
            })

    df = pd.DataFrame(rows).sort_values(["station_id", "timestamp"]).reset_index(drop=True)

    # True next-hour target within each station; final hour has no future label.
    df["next_hour_demand_kwh"] = (
        df.groupby("station_id")["energy_kwh"].shift(-1)
    )
    df = df.dropna(subset=["next_hour_demand_kwh"]).reset_index(drop=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    return df


if __name__ == "__main__":
    df = generate()
    print(f"Generated {len(df):,} rows -> {OUT}")
    print(f"Fault rate: {df['station_fault_flag'].mean():.2%}")
