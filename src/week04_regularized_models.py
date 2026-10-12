"""ChargeUp — SLM Week 4: Ridge/Lasso demand models + Logistic fault model.

Scope: feature engineering, Ridge, Lasso, Logistic Regression.
No tree/ensemble models are used here; those belong to later weeks.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, Lasso, LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/processed/chargeup_sessions_clean.csv'
OUT=ROOT/'reports/week04_regularized_models_metrics.json'

BASE=["temperature_c","humidity_pct","rainfall_mm","latitude","longitude","session_count","avg_session_minutes","avg_power_kw","connector_errors","voltage_v","current_a","availability_pct","energy_kwh"]
DERIVED=["hour_sin","hour_cos","dow_sin","dow_cos","is_weekend","is_peak_hour"]
FEATURES=BASE+DERIVED

def feature_frame(df: pd.DataFrame)->pd.DataFrame:
    x=df.copy(); ts=pd.to_datetime(x['timestamp'])
    h=ts.dt.hour; d=ts.dt.dayofweek
    x['hour_sin']=np.sin(2*np.pi*h/24); x['hour_cos']=np.cos(2*np.pi*h/24)
    x['dow_sin']=np.sin(2*np.pi*d/7); x['dow_cos']=np.cos(2*np.pi*d/7)
    x['is_weekend']=(d>=5).astype(int); x['is_peak_hour']=h.isin([7,8,9,17,18,19,20]).astype(int)
    return x

def split(df):
    df=df.sort_values(['timestamp','station_id']).reset_index(drop=True)
    times=np.array(sorted(pd.to_datetime(df.timestamp).unique())); cut=times[int(len(times)*.8)]
    train=df[pd.to_datetime(df.timestamp)<pd.Timestamp(cut)-pd.Timedelta(hours=1)].copy()
    test=df[pd.to_datetime(df.timestamp)>=pd.Timestamp(cut)].copy()
    return train,test,pd.Timestamp(cut)

def regm(y,p): return {'mae':float(mean_absolute_error(y,p)),'rmse':float(np.sqrt(mean_squared_error(y,p))),'r2':float(r2_score(y,p))}
def clfm(y,prob):
    pred=(prob>=.5).astype(int)
    return {'precision':float(precision_score(y,pred,zero_division=0)),'recall':float(recall_score(y,pred,zero_division=0)),'f1':float(f1_score(y,pred,zero_division=0)),'roc_auc':float(roc_auc_score(y,prob)),'pr_auc':float(average_precision_score(y,prob))}

def main():
    df=feature_frame(pd.read_csv(DATA,parse_dates=['timestamp']))
    tr,te,cut=split(df); Xtr=tr[FEATURES]; Xte=te[FEATURES]
    yr=tr.next_hour_demand_kwh; yt=te.next_hour_demand_kwh
    models={
      'ridge':Pipeline([('scale',StandardScaler()),('model',Ridge(alpha=3.0))]),
      'lasso':Pipeline([('scale',StandardScaler()),('model',Lasso(alpha=0.05,max_iter=10000,random_state=42))]),
    }
    reg={}
    for n,m in models.items(): m.fit(Xtr,yr); reg[n]=regm(yt,m.predict(Xte))
    logit=Pipeline([('scale',StandardScaler()),('model',LogisticRegression(max_iter=3000,class_weight='balanced',random_state=42))])
    logit.fit(Xtr,tr.station_fault_flag.astype(int)); prob=logit.predict_proba(Xte)[:,1]
    out={'week':4,'target_regression':'next_hour_demand_kwh','target_classification':'station_fault_flag','features':FEATURES,'engineered_features':DERIVED,'split_time':cut.isoformat(),'train_rows':len(tr),'test_rows':len(te),'regression':reg,'classification':{'logistic_regression':clfm(te.station_fault_flag.astype(int),prob)}}
    OUT.write_text(json.dumps(out,indent=2),encoding='utf-8'); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
