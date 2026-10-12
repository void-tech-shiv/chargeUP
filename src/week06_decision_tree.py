"""ChargeUp — SLM Week 6: interpretable Decision Tree for station faults."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import precision_score,recall_score,f1_score,roc_auc_score,average_precision_score
from sklearn.model_selection import TimeSeriesSplit
from .week04_regularized_models import feature_frame, FEATURES, split
ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/'data/processed/chargeup_sessions_clean.csv'; OUT=ROOT/'reports/week06_decision_tree_metrics.json'; SPLITS=ROOT/'reports/week06_top_splits.txt'

def metrics(y,p):
    pred=(p>=.5).astype(int)
    return {'precision':float(precision_score(y,pred,zero_division=0)),'recall':float(recall_score(y,pred,zero_division=0)),'f1':float(f1_score(y,pred,zero_division=0)),'roc_auc':float(roc_auc_score(y,p)),'pr_auc':float(average_precision_score(y,p))}

def main():
    df=feature_frame(pd.read_csv(DATA,parse_dates=['timestamp'])); tr,te,cut=split(df)
    Xtr=tr[FEATURES]; ytr=tr.station_fault_flag.astype(int); Xte=te[FEATURES]; yte=te.station_fault_flag.astype(int)
    cv=TimeSeriesSplit(n_splits=5); candidates=[]
    for depth in [2,3,4,5,6]:
      fold=[]; base=DecisionTreeClassifier(max_depth=depth,min_samples_leaf=12,class_weight='balanced',random_state=42)
      for a,b in cv.split(Xtr):
        if ytr.iloc[a].nunique()<2 or ytr.iloc[b].nunique()<2: continue
        m=clone(base); m.fit(Xtr.iloc[a],ytr.iloc[a]); fold.append(average_precision_score(ytr.iloc[b],m.predict_proba(Xtr.iloc[b])[:,1]))
      candidates.append({'max_depth':depth,'fold_pr_auc':[float(x) for x in fold],'mean_pr_auc':float(np.mean(fold))})
    best=max(candidates,key=lambda r:r['mean_pr_auc']); model=DecisionTreeClassifier(max_depth=best['max_depth'],min_samples_leaf=12,class_weight='balanced',random_state=42); model.fit(Xtr,ytr); prob=model.predict_proba(Xte)[:,1]
    imp=sorted([{'feature':f,'importance':float(v)} for f,v in zip(FEATURES,model.feature_importances_)],key=lambda x:x['importance'],reverse=True)
    text=export_text(model,feature_names=FEATURES,max_depth=3); SPLITS.write_text(text,encoding='utf-8')
    out={'week':6,'split_time':cut.isoformat(),'cv_candidates':candidates,'selected_max_depth':best['max_depth'],'test_metrics':metrics(yte,prob),'top_feature_importance':imp[:10],'top_splits_file':'reports/week06_top_splits.txt'}
    OUT.write_text(json.dumps(out,indent=2),encoding='utf-8'); print(json.dumps(out,indent=2)); print(text)
if __name__=='__main__': main()
