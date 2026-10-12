"""ChargeUp — SLM Week 9: K-Means, DBSCAN, PCA station segmentation and feedback feature."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import mean_absolute_error, average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from .clustering import fit_station_clusters
from .week04_regularized_models import feature_frame, FEATURES, split
ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/'data/processed/chargeup_sessions_clean.csv'; OUT=ROOT/'reports/week09_clustering_metrics.json'; SEG=ROOT/'reports/week09_station_segments.csv'; MAP=ROOT/'models/week09_station_segments.json'

def main():
    df=feature_frame(pd.read_csv(DATA,parse_dates=['timestamp'])); tr,te,cut=split(df)
    segment_map,station,art=fit_station_clusters(tr,n_clusters=4)
    station.to_csv(SEG,index=False); MAP.write_text(json.dumps({k:int(v) for k,v in segment_map.items()},indent=2),encoding='utf-8')
    for d in (tr,te): d['station_segment']=d.station_id.map(segment_map).fillna(-1).astype(float)
    X0=tr[FEATURES]; X1=tr[FEATURES+['station_segment']]; T0=te[FEATURES]; T1=te[FEATURES+['station_segment']]
    r0=Pipeline([('scale',StandardScaler()),('model',Ridge(alpha=3.0))]); r1=Pipeline([('scale',StandardScaler()),('model',Ridge(alpha=3.0))]); r0.fit(X0,tr.next_hour_demand_kwh); r1.fit(X1,tr.next_hour_demand_kwh)
    mae0=float(mean_absolute_error(te.next_hour_demand_kwh,r0.predict(T0))); mae1=float(mean_absolute_error(te.next_hour_demand_kwh,r1.predict(T1)))
    c0=Pipeline([('scale',StandardScaler()),('model',LogisticRegression(max_iter=3000,class_weight='balanced',random_state=42))]); c1=Pipeline([('scale',StandardScaler()),('model',LogisticRegression(max_iter=3000,class_weight='balanced',random_state=42))]); c0.fit(X0,tr.station_fault_flag); c1.fit(X1,tr.station_fault_flag)
    ap0=float(average_precision_score(te.station_fault_flag,c0.predict_proba(T0)[:,1])); ap1=float(average_precision_score(te.station_fault_flag,c1.predict_proba(T1)[:,1]))
    pca=art['pca']; db=station.dbscan_segment
    out={'week':9,'split_time':cut.isoformat(),'station_count':int(len(station)),'kmeans_clusters':int(station.kmeans_segment.nunique()),'dbscan_clusters_excluding_noise':int(db[db>=0].nunique()),'dbscan_noise_stations':int((db<0).sum()),'pca_explained_variance_ratio':[float(x) for x in pca.explained_variance_ratio_],'pca_total_explained_variance':float(pca.explained_variance_ratio_.sum()),'segment_feedback':{'ridge_mae_without_segment':mae0,'ridge_mae_with_segment':mae1,'logistic_pr_auc_without_segment':ap0,'logistic_pr_auc_with_segment':ap1},'station_report':'reports/week09_station_segments.csv','segment_map':'models/week09_station_segments.json'}
    OUT.write_text(json.dumps(out,indent=2),encoding='utf-8'); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
