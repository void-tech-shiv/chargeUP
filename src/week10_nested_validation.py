"""ChargeUp — SLM Week 10: nested CV, calibration, imbalance metrics, significance tests."""
from __future__ import annotations
import json
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, balanced_accuracy_score, matthews_corrcoef, brier_score_loss
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor, XGBClassifier
from .week04_regularized_models import feature_frame, FEATURES
ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/'data/processed/chargeup_sessions_clean.csv'; OUT=ROOT/'reports/week10_validation_metrics.json'

def reg_metrics(y,p): return {'mae':float(mean_absolute_error(y,p)),'rmse':float(np.sqrt(mean_squared_error(y,p))),'r2':float(r2_score(y,p))}
def clf_metrics(y,p):
  pred=(p>=.5).astype(int); return {'precision':float(precision_score(y,pred,zero_division=0)),'recall':float(recall_score(y,pred,zero_division=0)),'f1':float(f1_score(y,pred,zero_division=0)),'roc_auc':float(roc_auc_score(y,p)),'pr_auc':float(average_precision_score(y,p)),'balanced_accuracy':float(balanced_accuracy_score(y,pred)),'mcc':float(matthews_corrcoef(y,pred)),'brier':float(brier_score_loss(y,p))}

def xgb_reg(depth): return XGBRegressor(n_estimators=220,learning_rate=.04,max_depth=depth,subsample=.9,colsample_bytree=.9,objective='reg:squarederror',eval_metric='rmse',random_state=42,n_jobs=4)
def xgb_clf(depth,spw): return XGBClassifier(n_estimators=180,learning_rate=.04,max_depth=depth,subsample=.9,colsample_bytree=.9,objective='binary:logistic',eval_metric='aucpr',scale_pos_weight=spw,random_state=42,n_jobs=4)

def nested_reg(X,y):
  outer=TimeSeriesSplit(n_splits=3); rows=[]; choices=[]
  for oi,(tr,va) in enumerate(outer.split(X),1):
    inner=TimeSeriesSplit(n_splits=2); cand=[]
    for depth in [3,5]:
      sc=[]
      for a,b in inner.split(X.iloc[tr]):
        m=xgb_reg(depth); m.fit(X.iloc[tr].iloc[a],y.iloc[tr].iloc[a]); sc.append(mean_absolute_error(y.iloc[tr].iloc[b],m.predict(X.iloc[tr].iloc[b])))
      cand.append((float(np.mean(sc)),depth))
    _,best=min(cand); choices.append(best); m=xgb_reg(best); m.fit(X.iloc[tr],y.iloc[tr]); pred=m.predict(X.iloc[va]); rows.append({'outer_fold':oi,'best_max_depth':best,'mae':float(mean_absolute_error(y.iloc[va],pred))})
  return rows,Counter(choices).most_common(1)[0][0]

def nested_clf(X,y):
  outer=TimeSeriesSplit(n_splits=3); rows=[]; choices=[]; pos=max(int(y.sum()),1); neg=max(len(y)-pos,1); spw=neg/pos
  for oi,(tr,va) in enumerate(outer.split(X),1):
    inner=TimeSeriesSplit(n_splits=2); cand=[]
    for depth in [3,4]:
      sc=[]
      for a,b in inner.split(X.iloc[tr]):
        ya=y.iloc[tr].iloc[a]; yb=y.iloc[tr].iloc[b]
        if ya.nunique()<2 or yb.nunique()<2: continue
        m=xgb_clf(depth,spw); m.fit(X.iloc[tr].iloc[a],ya); sc.append(average_precision_score(yb,m.predict_proba(X.iloc[tr].iloc[b])[:,1]))
      cand.append((float(np.mean(sc)) if sc else -1,depth))
    _,best=max(cand); choices.append(best); m=xgb_clf(best,spw); m.fit(X.iloc[tr],y.iloc[tr]); prob=m.predict_proba(X.iloc[va])[:,1]; rows.append({'outer_fold':oi,'best_max_depth':best,'pr_auc':float(average_precision_score(y.iloc[va],prob))})
  return rows,Counter(choices).most_common(1)[0][0],spw

def bootstrap_regression_significance(y,champ,base,n=2000):
  rng=np.random.default_rng(42); d=np.abs(y-champ)-np.abs(y-base); boots=[]
  for _ in range(n): boots.append(float(np.mean(d[rng.integers(0,len(d),len(d))])))
  lo,hi=np.quantile(boots,[.025,.975]); p=2*min(np.mean(np.array(boots)>=0),np.mean(np.array(boots)<=0))
  return {'mean_absolute_error_difference_champion_minus_ols':float(np.mean(d)),'bootstrap_95_ci':[float(lo),float(hi)],'two_sided_bootstrap_p_value':float(min(p,1.0))}

def main():
  df=feature_frame(pd.read_csv(DATA,parse_dates=['timestamp'])).sort_values(['timestamp','station_id']).reset_index(drop=True); times=np.array(sorted(df.timestamp.unique())); cut=times[int(len(times)*.8)]
  dev=df[df.timestamp<pd.Timestamp(cut)-pd.Timedelta(hours=1)].copy().reset_index(drop=True); test=df[df.timestamp>=pd.Timestamp(cut)].copy().reset_index(drop=True)
  X=dev[FEATURES]; Xt=test[FEATURES]; yr=dev.next_hour_demand_kwh.astype(float); yrt=test.next_hour_demand_kwh.astype(float); yc=dev.station_fault_flag.astype(int); yct=test.station_fault_flag.astype(int)
  reg_outer,reg_depth=nested_reg(X,yr); clf_outer,clf_depth,spw=nested_clf(X,yc)
  reg=xgb_reg(reg_depth); reg.fit(X,yr); rp=reg.predict(Xt)
  ols=Pipeline([('scale',StandardScaler()),('model',LinearRegression())]); ols.fit(X,yr); op=ols.predict(Xt)
  base_clf=xgb_clf(clf_depth,spw)
  calibrated=CalibratedClassifierCV(estimator=base_clf,method='sigmoid',cv=TimeSeriesSplit(n_splits=3)); calibrated.fit(X,yc); cp=calibrated.predict_proba(Xt)[:,1]
  logit=Pipeline([('scale',StandardScaler()),('model',LogisticRegression(max_iter=3000,class_weight='balanced',random_state=42))]); logit.fit(X,yc); lp=logit.predict_proba(Xt)[:,1]
  c_pred=(cp>=.5).astype(int); l_pred=(lp>=.5).astype(int); b=int(np.sum((c_pred==yct)&(l_pred!=yct))); c=int(np.sum((c_pred!=yct)&(l_pred==yct))); mcn=float(binomtest(min(b,c),n=b+c,p=.5,alternative='two-sided').pvalue) if b+c else 1.0
  out={'week':10,'sealed_test_start':pd.Timestamp(cut).isoformat(),'development_rows':len(dev),'sealed_test_rows':len(test),'nested_cv':{'regression_outer':reg_outer,'selected_regression_max_depth':reg_depth,'classification_outer':clf_outer,'selected_classification_max_depth':clf_depth},'sealed_test':{'xgboost_regression':reg_metrics(yrt,rp),'ols_reference':reg_metrics(yrt,op),'calibrated_xgboost_classification':clf_metrics(yct,cp),'logistic_reference':clf_metrics(yct,lp)},'significance_tests':{'regression_paired_bootstrap':bootstrap_regression_significance(yrt.to_numpy(),rp,op),'classification_mcnemar_exact':{'champion_correct_logistic_wrong':b,'champion_wrong_logistic_correct':c,'p_value':mcn}},'calibration':'sigmoid via CalibratedClassifierCV with TimeSeriesSplit(3)','imbalance_metric_priority':'PR-AUC plus balanced accuracy/MCC/Brier'}
  OUT.write_text(json.dumps(out,indent=2),encoding='utf-8'); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
