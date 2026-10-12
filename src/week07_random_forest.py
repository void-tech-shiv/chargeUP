"""ChargeUp — SLM Week 7: Random Forest, OOB error, permutation importance."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import precision_score,recall_score,f1_score,roc_auc_score,average_precision_score
from .week04_regularized_models import feature_frame, FEATURES, split
ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/'data/processed/chargeup_sessions_clean.csv'; OUT=ROOT/'reports/week07_random_forest_metrics.json'; IMP=ROOT/'reports/week07_permutation_importance.csv'

def metrics(y,p):
  pred=(p>=.5).astype(int)
  return {'precision':float(precision_score(y,pred,zero_division=0)),'recall':float(recall_score(y,pred,zero_division=0)),'f1':float(f1_score(y,pred,zero_division=0)),'roc_auc':float(roc_auc_score(y,p)),'pr_auc':float(average_precision_score(y,p))}

def main():
  df=feature_frame(pd.read_csv(DATA,parse_dates=['timestamp'])); tr,te,cut=split(df); Xtr=tr[FEATURES]; ytr=tr.station_fault_flag.astype(int); Xte=te[FEATURES]; yte=te.station_fault_flag.astype(int)
  model=RandomForestClassifier(n_estimators=350,max_depth=12,min_samples_leaf=3,class_weight='balanced_subsample',oob_score=True,random_state=42,n_jobs=-1)
  model.fit(Xtr,ytr); prob=model.predict_proba(Xte)[:,1]
  perm=permutation_importance(model,Xte,yte,n_repeats=12,random_state=42,scoring='average_precision',n_jobs=-1)
  imp=pd.DataFrame({'feature':FEATURES,'pr_auc_drop_mean':perm.importances_mean,'pr_auc_drop_std':perm.importances_std}).sort_values('pr_auc_drop_mean',ascending=False); imp.to_csv(IMP,index=False)
  out={'week':7,'split_time':cut.isoformat(),'oob_score':float(model.oob_score_),'oob_error':float(1-model.oob_score_),'test_metrics':metrics(yte,prob),'top_permutation_importance':imp.head(10).to_dict(orient='records'),'importance_file':'reports/week07_permutation_importance.csv'}
  OUT.write_text(json.dumps(out,indent=2),encoding='utf-8'); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
