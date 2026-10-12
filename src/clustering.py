from __future__ import annotations
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]

CLUSTER_COLS = [
    "energy_kwh",
    "session_count",
    "avg_session_minutes",
    "avg_power_kw",
    "connector_errors",
    "availability_pct",
]


def fit_station_clusters(df: pd.DataFrame, n_clusters: int = 4):
    station = (
        df.groupby("station_id")[CLUSTER_COLS]
        .mean()
        .sort_index()
        .copy()
    )

    scaler = StandardScaler()
    Xs = scaler.fit_transform(station)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    station["kmeans_segment"] = kmeans.fit_predict(Xs)

    dbscan = DBSCAN(eps=1.5, min_samples=2)
    station["dbscan_segment"] = dbscan.fit_predict(Xs)

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(Xs)
    station["pca_1"] = coords[:, 0]
    station["pca_2"] = coords[:, 1]

    segment_map = station["kmeans_segment"].astype(int).to_dict()

    artifacts = {
        "scaler": scaler,
        "kmeans": kmeans,
        "dbscan": dbscan,
        "pca": pca,
        "cluster_columns": CLUSTER_COLS,
    }
    return segment_map, station.reset_index(), artifacts


def save_station_clusters(segment_map, station_report, artifacts):
    models = ROOT / "models"
    reports = ROOT / "reports"
    models.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)

    (models / "station_segments.json").write_text(
        json.dumps(segment_map, indent=2), encoding="utf-8"
    )
    joblib.dump(artifacts, models / "station_cluster_models.joblib")
    station_report.to_csv(reports / "station_segments.csv", index=False)
