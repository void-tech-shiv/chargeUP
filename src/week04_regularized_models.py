"""SLM Week 4 on real Palo Alto data: Ridge, Lasso, Logistic Regression."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge,Lasso,LogisticRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score,precision_score,recall_score,f1_score,roc_auc_score,average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from .palo_alto_schema import BASE_FEATURES,REGRESSION_TARGET,CLASSIFICATION_TARGET,FAULT_PROXY_DESCRIPTION

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/processed/chargeup_sessions_clean.csv'
OUT=ROOT/'reports/week04_regularized_models_metrics.json'
DERIVED=['hour_sin','hour_cos','dow_sin','dow_cos','is_weekend','is_peak_hour']
FEATURES=BASE_FEATURES+DERIVED

def feature_frame(df: pd.DataFrame)->pd.DataFrame:
    x=df.copy(); x['timestamp']=pd.to_datetime(x['timestamp'],errors='coerce'); h=x.timestamp.dt.hour.fillna(0); d=x.timestamp.dt.dayofweek.fillna(0)
    x['hour_sin']=np.sin(2*np.pi*h/24); x['hour_cos']=np.cos(2*np.pi*h/24); x['dow_sin']=np.sin(2*np.pi*d/7); x['dow_cos']=np.cos(2*np.pi*d/7); x['is_weekend']=(d>=5).astype(int); x['is_peak_hour']=h.isin([7,8,9,17,18,19,20]).astype(int)
    x[FEATURES]=x[FEATURES].apply(pd.to_numeric,errors='coerce').replace([np.inf,-np.inf],np.nan).fillna(0.0)
    return x

def split(df):
    df=df.sort_values(['timestamp','station_id']).reset_index(drop=True); times=np.array(sorted(pd.to_datetime(df.timestamp).unique())); cut=pd.Timestamp(times[int(len(times)*.8)])
    train=df[pd.to_datetime(df.timestamp)<cut-pd.Timedelta(hours=1)].copy(); test=df[pd.to_datetime(df.timestamp)>=cut].copy(); return train,test,cut

def regm(y,p): return {'mae':float(mean_absolute_error(y,p)),'rmse':float(np.sqrt(mean_squared_error(y,p))),'r2':float(r2_score(y,p))}
def clfm(y,prob):
    pred=(prob>=.5).astype(int); out={'precision':float(precision_score(y,pred,zero_division=0)),'recall':float(recall_score(y,pred,zero_division=0)),'f1':float(f1_score(y,pred,zero_division=0)),'positive_rate':float(np.mean(y))}
    if len(np.unique(y))>1:
        out['roc_auc']=float(roc_auc_score(y,prob)); out['pr_auc']=float(average_precision_score(y,prob))
    else:
        out['roc_auc']=None; out['pr_auc']=None
    return out

def main():
    df=feature_frame(pd.read_csv(DATA,parse_dates=['timestamp'])); tr,te,cut=split(df); Xtr=tr[FEATURES]; Xte=te[FEATURES]; yr=tr[REGRESSION_TARGET].astype(float); yt=te[REGRESSION_TARGET].astype(float)
    models={'ridge':Pipeline([('scale',StandardScaler()),('model',Ridge(alpha=3.0))]),'lasso':Pipeline([('scale',StandardScaler()),('model',Lasso(alpha=.05,max_iter=10000,random_state=42))])}
    reg={}
    for n,m in models.items(): m.fit(Xtr,yr); reg[n]=regm(yt,m.predict(Xte))
    yc=tr[CLASSIFICATION_TARGET].astype(int); yct=te[CLASSIFICATION_TARGET].astype(int)
    if yc.nunique()<2: raise ValueError('Training split has no positive server-termination proxy examples.')
    logit=Pipeline([('scale',StandardScaler()),('model',LogisticRegression(max_iter=3000,class_weight='balanced',random_state=42))]); logit.fit(Xtr,yc); prob=logit.predict_proba(Xte)[:,1]
    out={'week':4,'dataset':'City of Palo Alto ChargePoint CY20Q4','target_regression':REGRESSION_TARGET,'target_classification':CLASSIFICATION_TARGET,'classification_target_note':FAULT_PROXY_DESCRIPTION,'features':FEATURES,'engineered_features':DERIVED,'split_time':cut.isoformat(),'train_rows':len(tr),'test_rows':len(te),'regression':reg,'classification':{'logistic_regression':clfm(yct,prob)}}
    OUT.write_text(json.dumps(out,indent=2),encoding='utf-8'); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
