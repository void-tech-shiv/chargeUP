"""ChargeUp — SLM Week 5: time-series CV and one-standard-error selection."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LinearRegression,Ridge,Lasso,LogisticRegression
from sklearn.metrics import mean_absolute_error,average_precision_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from .week04_regularized_models import feature_frame, FEATURES
ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/'data/processed/chargeup_sessions_clean.csv'; OUT=ROOT/'reports/week05_cv_selection.json'

def reg_candidates():
    return [
      ('linear',0,Pipeline([('scale',StandardScaler()),('model',LinearRegression())])),
      ('ridge_a1',1,Pipeline([('scale',StandardScaler()),('model',Ridge(alpha=1.0))])),
      ('ridge_a10',2,Pipeline([('scale',StandardScaler()),('model',Ridge(alpha=10.0))])),
      ('lasso_a0.02',3,Pipeline([('scale',StandardScaler()),('model',Lasso(alpha=.02,max_iter=10000,random_state=42))])),
      ('lasso_a0.10',4,Pipeline([('scale',StandardScaler()),('model',Lasso(alpha=.10,max_iter=10000,random_state=42))])),
    ]

def main():
    df=feature_frame(pd.read_csv(DATA,parse_dates=['timestamp'])).sort_values(['timestamp','station_id']).reset_index(drop=True)
    times=np.array(sorted(df.timestamp.unique())); cut=times[int(len(times)*.8)]
    dev=df[df.timestamp<pd.Timestamp(cut)-pd.Timedelta(hours=1)].copy().reset_index(drop=True)
    X=dev[FEATURES]; y=dev.next_hour_demand_kwh.astype(float); tscv=TimeSeriesSplit(n_splits=5)
    rows=[]
    for name,complexity,model in reg_candidates():
      scores=[]
      for tr,va in tscv.split(X):
        m=clone(model); m.fit(X.iloc[tr],y.iloc[tr]); scores.append(mean_absolute_error(y.iloc[va],m.predict(X.iloc[va])))
      rows.append({'name':name,'complexity_rank':complexity,'fold_mae':[float(x) for x in scores],'mean_mae':float(np.mean(scores)),'se_mae':float(np.std(scores,ddof=1)/np.sqrt(len(scores)))})
    best=min(rows,key=lambda r:r['mean_mae']); threshold=best['mean_mae']+best['se_mae']
    eligible=[r for r in rows if r['mean_mae']<=threshold]
    selected=min(eligible,key=lambda r:r['complexity_rank'])
    clf=Pipeline([('scale',StandardScaler()),('model',LogisticRegression(max_iter=3000,class_weight='balanced',random_state=42))])
    pr=[]; yc=dev.station_fault_flag.astype(int)
    for tr,va in tscv.split(X):
      if yc.iloc[tr].nunique()<2 or yc.iloc[va].nunique()<2: continue
      m=clone(clf); m.fit(X.iloc[tr],yc.iloc[tr]); pr.append(average_precision_score(yc.iloc[va],m.predict_proba(X.iloc[va])[:,1]))
    out={'week':5,'cv':'TimeSeriesSplit(n_splits=5)','sealed_holdout_starts':pd.Timestamp(cut).isoformat(),'regression_candidates':rows,'one_standard_error_rule':{'best_mean_model':best['name'],'threshold_mae':float(threshold),'eligible_models':[r['name'] for r in eligible],'selected_simplest_model':selected['name']},'logistic_regression_baseline':{'fold_pr_auc':[float(x) for x in pr],'mean_pr_auc':float(np.mean(pr)),'se_pr_auc':float(np.std(pr,ddof=1)/np.sqrt(len(pr)))},'milestone':'linear/logistic baseline selection complete'}
    OUT.write_text(json.dumps(out,indent=2),encoding='utf-8'); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
