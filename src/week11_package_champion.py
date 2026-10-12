"""ChargeUp — SLM Week 11: versioned champion artifact + PSI baseline."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from .clustering import fit_station_clusters
from .drift import make_baseline_stats
from .model_features import build_model_features, MODEL_FEATURES
ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/'data/processed/chargeup_sessions_clean.csv'; VERSION='chargeup-v1.0.0'; DIR=ROOT/'models/registry'/VERSION

def sha(path):
    h=hashlib.sha256(); h.update(path.read_bytes()); return h.hexdigest()
def main():
    DIR.mkdir(parents=True,exist_ok=True); df=pd.read_csv(DATA,parse_dates=['timestamp']).sort_values(['timestamp','station_id']).reset_index(drop=True)
    seg,station,_=fit_station_clusters(df); X=build_model_features(df,seg)
    reg=XGBRegressor(n_estimators=220,learning_rate=.04,max_depth=3,subsample=.9,colsample_bytree=.9,objective='reg:squarederror',eval_metric='rmse',random_state=42,n_jobs=4); reg.fit(X,df.next_hour_demand_kwh.astype(float))
    base=Pipeline([('scale',StandardScaler()),('model',LogisticRegression(max_iter=3000,class_weight='balanced',random_state=42))]); clf=CalibratedClassifierCV(estimator=base,method='sigmoid',cv=TimeSeriesSplit(n_splits=3)); clf.fit(X,df.station_fault_flag.astype(int))
    rp=DIR/'regression_model.joblib'; fp=DIR/'fault_model.joblib'; joblib.dump(reg,rp); joblib.dump(clf,fp)
    (DIR/'station_segments.json').write_text(json.dumps({k:int(v) for k,v in seg.items()},indent=2),encoding='utf-8'); (DIR/'baseline_stats.json').write_text(json.dumps(make_baseline_stats(X),indent=2),encoding='utf-8')
    schema={'version':VERSION,'features':MODEL_FEATURES,'targets':{'regression':'next_hour_demand_kwh','classification':'station_fault_flag'},'feature_builder':'src.model_features.build_model_features'}; (DIR/'feature_schema.json').write_text(json.dumps(schema,indent=2),encoding='utf-8')
    metrics=json.loads((ROOT/'reports/week10_validation_metrics.json').read_text())
    manifest={'version':VERSION,'slm_week':11,'regression_champion':'xgboost_max_depth_3','classification_champion':'calibrated_logistic_regression','feature_count':len(MODEL_FEATURES),'sealed_test_metrics':{'regression':metrics['sealed_test']['xgboost_regression'],'classification':metrics['sealed_test']['logistic_reference']},'psi_baseline':'baseline_stats.json','artifacts':{'regression_model':'regression_model.joblib','fault_model':'fault_model.joblib','station_segments':'station_segments.json','feature_schema':'feature_schema.json'},'sha256':{'regression_model.joblib':sha(rp),'fault_model.joblib':sha(fp)},'retraining_trigger':'PSI >= 0.25'}
    (DIR/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    (DIR/'README.md').write_text('# ChargeUp v1.0.0\n\nRun `python -m src.week11_package_champion` to regenerate the versioned `.joblib` artifacts. Text manifests, feature schema, station segments, and PSI baseline are committed; binary model files are build artifacts.\n',encoding='utf-8')
    print(json.dumps(manifest,indent=2))
if __name__=='__main__': main()
