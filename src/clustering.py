from __future__ import annotations
import json
from pathlib import Path
import joblib
import pandas as pd
from sklearn.cluster import KMeans,DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from .palo_alto_schema import CLUSTER_FEATURES

ROOT=Path(__file__).resolve().parents[1]
CLUSTER_COLS=CLUSTER_FEATURES

def fit_station_clusters(df: pd.DataFrame,n_clusters:int=4):
    cols=[c for c in CLUSTER_COLS if c in df.columns]
    if not cols: raise ValueError('No Palo Alto clustering features are present.')
    station=df.groupby('station_id')[cols].mean().sort_index().copy().fillna(0.0)
    scaler=StandardScaler(); Xs=scaler.fit_transform(station)
    k=max(2,min(n_clusters,len(station))) if len(station)>1 else 1
    kmeans=KMeans(n_clusters=k,random_state=42,n_init=20); station['kmeans_segment']=kmeans.fit_predict(Xs)
    dbscan=DBSCAN(eps=1.5,min_samples=2); station['dbscan_segment']=dbscan.fit_predict(Xs)
    pca=PCA(n_components=min(2,Xs.shape[1],len(station)),random_state=42); coords=pca.fit_transform(Xs)
    station['pca_1']=coords[:,0]; station['pca_2']=coords[:,1] if coords.shape[1]>1 else 0.0
    segment_map=station['kmeans_segment'].astype(int).to_dict()
    artifacts={'scaler':scaler,'kmeans':kmeans,'dbscan':dbscan,'pca':pca,'cluster_columns':cols}
    return segment_map,station.reset_index(),artifacts

def save_station_clusters(segment_map,station_report,artifacts):
    models=ROOT/'models'; reports=ROOT/'reports'; models.mkdir(parents=True,exist_ok=True); reports.mkdir(parents=True,exist_ok=True)
    (models/'station_segments.json').write_text(json.dumps({k:int(v) for k,v in segment_map.items()},indent=2),encoding='utf-8'); joblib.dump(artifacts,models/'station_cluster_models.joblib'); station_report.to_csv(reports/'station_segments.csv',index=False)
