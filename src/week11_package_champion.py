"""SLM Week 11: package champions selected in Week 10 on real Palo Alto data."""
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
from .model_features import build_model_features,MODEL_FEATURES
from .palo_alto_schema import FAULT_PROXY_DESCRIPTION,REGRESSION_TARGET,CLASSIFICATION_TARGET

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/processed/chargeup_sessions_clean.csv'
METRICS=ROOT/'reports/week10_validation_metrics.json'
VERSION='chargeup-paloalto-v2.0.0'
DIR=ROOT/'models/registry'/VERSION

def sha(path):
    h=hashlib.sha256(); h.update(path.read_bytes()); return h.hexdigest()

def main():
    if not METRICS.exists():
        raise FileNotFoundError('Run Week 10 before packaging the Week 11 champion.')
    metrics=json.loads(METRICS.read_text(encoding='utf-8'))
    reg_depth=int(metrics.get('nested_cv',{}).get('selected_regression_max_depth',3))
    sealed=metrics.get('sealed_test',{})

    DIR.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(DATA,parse_dates=['timestamp']).sort_values(['timestamp','station_id']).reset_index(drop=True)
    seg,_,_=fit_station_clusters(df); X=build_model_features(df,seg)

    # Week 10 selects XGBoost depth with nested temporal CV. Week 11 packages
    # that exact selected configuration rather than hard-coding a different model.
    reg=XGBRegressor(
        n_estimators=260,learning_rate=.04,max_depth=reg_depth,
        subsample=.9,colsample_bytree=.9,objective='reg:squarederror',
        eval_metric='rmse',random_state=42,n_jobs=4,
    )
    reg.fit(X,df[REGRESSION_TARGET].astype(float))

    # On the sealed real-data test set, the logistic reference has the stronger
    # PR-AUC than calibrated XGBoost, so it is the classification champion.
    logistic_metrics=sealed.get('logistic_reference') or {}
    xgb_metrics=sealed.get('calibrated_xgboost_classification') or {}
    if (
        logistic_metrics.get('pr_auc') is not None
        and xgb_metrics.get('pr_auc') is not None
        and float(logistic_metrics['pr_auc']) < float(xgb_metrics['pr_auc'])
    ):
        raise RuntimeError(
            'Week 10 no longer selects logistic regression by PR-AUC; '
            'update Week 11 packaging before releasing a mismatched classifier.'
        )
    base=Pipeline([
        ('scale',StandardScaler()),
        ('model',LogisticRegression(max_iter=3000,class_weight='balanced',random_state=42)),
    ])
    clf=CalibratedClassifierCV(estimator=base,method='sigmoid',cv=TimeSeriesSplit(n_splits=3))
    clf.fit(X,df[CLASSIFICATION_TARGET].astype(int))

    rp=DIR/'regression_model.joblib'; fp=DIR/'fault_model.joblib'
    joblib.dump(reg,rp); joblib.dump(clf,fp)
    (DIR/'station_segments.json').write_text(json.dumps({k:int(v) for k,v in seg.items()},indent=2),encoding='utf-8')
    (DIR/'baseline_stats.json').write_text(json.dumps(make_baseline_stats(X),indent=2),encoding='utf-8')
    schema={
        'version':VERSION,
        'dataset':'City of Palo Alto ChargePoint Data CY20Q4.csv',
        'features':MODEL_FEATURES,
        'targets':{'regression':REGRESSION_TARGET,'classification':CLASSIFICATION_TARGET},
        'classification_target_note':FAULT_PROXY_DESCRIPTION,
        'feature_builder':'src.model_features.build_model_features',
        'week10_selected_regression_max_depth':reg_depth,
    }
    (DIR/'feature_schema.json').write_text(json.dumps(schema,indent=2),encoding='utf-8')

    manifest={
        'version':VERSION,
        'slm_week':11,
        'dataset':'City of Palo Alto Electric Vehicle Charging Station Usage (July 2011 - Dec 2020)',
        'source_file':'ChargePoint Data CY20Q4.csv',
        'selection_source':'reports/week10_validation_metrics.json',
        'regression_champion':f'xgboost_max_depth_{reg_depth}',
        'classification_champion':'calibrated_logistic_regression',
        'classification_champion_rule':'higher sealed-test PR-AUC than calibrated XGBoost',
        'classification_target_note':FAULT_PROXY_DESCRIPTION,
        'feature_count':len(MODEL_FEATURES),
        'sealed_test_metrics':{
            'regression':sealed.get('xgboost_regression'),
            'classification':logistic_metrics,
        },
        'psi_baseline':'baseline_stats.json',
        'artifacts':{
            'regression_model':'regression_model.joblib',
            'fault_model':'fault_model.joblib',
            'station_segments':'station_segments.json',
            'feature_schema':'feature_schema.json',
        },
        'sha256':{
            'regression_model.joblib':sha(rp),
            'fault_model.joblib':sha(fp),
        },
        'retraining_trigger':'PSI >= 0.25',
    }
    (DIR/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    (DIR/'README.md').write_text(
        '# ChargeUp Palo Alto v2.0.0\n\n'
        'Models are trained on the City of Palo Alto ChargePoint dataset. '
        'Week 11 packages the exact regression depth selected by Week 10 nested CV. '
        'The classification target is a documented next-hour CPS/server-termination '
        'proxy because the source CSV has no ground-truth hardware-failure label.\n',
        encoding='utf-8',
    )
    print(json.dumps(manifest,indent=2))

if __name__=='__main__': main()
