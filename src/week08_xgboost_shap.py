"""ChargeUp — SLM Week 8: XGBoost champions, early stopping, SHAP."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import shap
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score,precision_score,recall_score,f1_score,roc_auc_score,average_precision_score
from .week04_regularized_models import feature_frame, FEATURES
ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/'data/processed/chargeup_sessions_clean.csv'; OUT=ROOT/'reports/week08_xgboost_metrics.json'; SHAP_OUT=ROOT/'reports/week08_shap_importance.csv'

def temporal_parts(df):
    df=df.sort_values(['timestamp','station_id']).reset_index(drop=True); times=np.array(sorted(df.timestamp.unique())); n=len(times)
    t1=times[int(n*.70)]; t2=times[int(n*.80)]
    train=df[df.timestamp<pd.Timestamp(t1)-pd.Timedelta(hours=1)].copy(); val=df[(df.timestamp>=pd.Timestamp(t1))&(df.timestamp<pd.Timestamp(t2)-pd.Timedelta(hours=1))].copy(); test=df[df.timestamp>=pd.Timestamp(t2)].copy()
    return train,val,test,pd.Timestamp(t1),pd.Timestamp(t2)
def regm(y,p): return {'mae':float(mean_absolute_error(y,p)),'rmse':float(np.sqrt(mean_squared_error(y,p))),'r2':float(r2_score(y,p))}
def clfm(y,p):
    pr=(p>=.5).astype(int); return {'precision':float(precision_score(y,pr,zero_division=0)),'recall':float(recall_score(y,pr,zero_division=0)),'f1':float(f1_score(y,pr,zero_division=0)),'roc_auc':float(roc_auc_score(y,p)),'pr_auc':float(average_precision_score(y,p))}

def main():
    df=feature_frame(pd.read_csv(DATA,parse_dates=['timestamp'])); tr,va,te,t1,t2=temporal_parts(df)
    Xtr=tr[FEATURES]; Xv=va[FEATURES]; Xt=te[FEATURES]
    reg=XGBRegressor(n_estimators=1600,learning_rate=.03,max_depth=5,subsample=.9,colsample_bytree=.9,objective='reg:squarederror',eval_metric='rmse',early_stopping_rounds=60,random_state=42,n_jobs=4)
    reg.fit(Xtr,tr.next_hour_demand_kwh,eval_set=[(Xv,va.next_hour_demand_kwh)],verbose=False)
    rp=reg.predict(Xt)
    ytr=tr.station_fault_flag.astype(int); yv=va.station_fault_flag.astype(int); yt=te.station_fault_flag.astype(int); pos=max(int(ytr.sum()),1); neg=max(len(ytr)-pos,1)
    clf=XGBClassifier(n_estimators=1600,learning_rate=.03,max_depth=4,subsample=.9,colsample_bytree=.9,objective='binary:logistic',eval_metric='aucpr',early_stopping_rounds=60,scale_pos_weight=neg/pos,random_state=42,n_jobs=4)
    clf.fit(Xtr,ytr,eval_set=[(Xv,yv)],verbose=False); cp=clf.predict_proba(Xt)[:,1]
    sample=Xt.iloc[:min(500,len(Xt))]
    reg_ex=shap.TreeExplainer(reg); reg_sv=np.asarray(reg_ex.shap_values(sample)); reg_imp=np.mean(np.abs(reg_sv),axis=0)
    clf_ex=shap.TreeExplainer(clf); clf_sv=np.asarray(clf_ex.shap_values(sample)); clf_imp=np.mean(np.abs(clf_sv),axis=0)
    sh=pd.DataFrame({'feature':FEATURES,'regression_mean_abs_shap':reg_imp,'classification_mean_abs_shap':clf_imp}).sort_values('regression_mean_abs_shap',ascending=False); sh.to_csv(SHAP_OUT,index=False)
    out={'week':8,'train_end':t1.isoformat(),'sealed_test_start':t2.isoformat(),'train_rows':len(tr),'validation_rows':len(va),'test_rows':len(te),'regression':{'best_iteration':int(reg.best_iteration),'metrics':regm(te.next_hour_demand_kwh,rp)},'classification':{'best_iteration':int(clf.best_iteration),'scale_pos_weight':float(neg/pos),'metrics':clfm(yt,cp)},'shap_file':'reports/week08_shap_importance.csv','top_shap':sh.head(10).to_dict(orient='records')}
    OUT.write_text(json.dumps(out,indent=2),encoding='utf-8'); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
