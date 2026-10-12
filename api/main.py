from __future__ import annotations
import json
from datetime import datetime,timezone
from pathlib import Path
from typing import List
import joblib
import pandas as pd
from fastapi import FastAPI,HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel,Field
from src.model_features import build_model_features
from src.drift import check_drift

ROOT=Path(__file__).resolve().parents[1]
VERSION='chargeup-paloalto-v2.0.0'
MODELS=ROOT/'models'/'registry'/VERSION
DATA=ROOT/'data'/'processed'/'chargeup_sessions_clean.csv'
STATIC=ROOT/'api'/'static'
app=FastAPI(title='ChargeUp API',version='2.0.0',description='Palo Alto next-hour EV charging demand + server-termination proxy forecasting.')
app.mount('/static',StaticFiles(directory=STATIC),name='static')
_runtime={'prediction_requests':0,'drift_checks':0,'retrain_requests':0,'started_at':datetime.now(timezone.utc).isoformat()}
reg_model=fault_model=None; station_segments={}; baseline_stats={}; model_meta={}

def load_artifacts():
    global reg_model,fault_model,station_segments,baseline_stats,model_meta
    req=[MODELS/'regression_model.joblib',MODELS/'fault_model.joblib',MODELS/'station_segments.json',MODELS/'baseline_stats.json',MODELS/'manifest.json']
    missing=[str(x) for x in req if not x.exists()]
    if missing: raise RuntimeError('Missing Palo Alto v2 artifacts. Run the real-data SLM pipeline first:\n'+'\n'.join(missing))
    reg_model=joblib.load(MODELS/'regression_model.joblib'); fault_model=joblib.load(MODELS/'fault_model.joblib')
    station_segments=json.loads((MODELS/'station_segments.json').read_text()); baseline_stats=json.loads((MODELS/'baseline_stats.json').read_text()); model_meta=json.loads((MODELS/'manifest.json').read_text())
load_artifacts()

class ChargeUpInput(BaseModel):
    station_id:str='PALO ALTO CA / HAMILTON #1'
    timestamp:datetime=Field(default_factory=datetime.now)
    latitude:float=37.444572
    longitude:float=-122.160309
    session_count:int=1
    avg_session_minutes:float=90.0
    avg_total_duration_minutes:float=120.0
    avg_power_kw:float=6.2
    energy_kwh:float=8.0
    fee_usd:float=0.0
    level1_share:float=0.0
    level2_share:float=1.0
    dc_fast_share:float=0.0

def predict_one(item:ChargeUpInput)->dict:
    X=build_model_features(pd.DataFrame([item.model_dump()]),station_segments)
    demand=max(0.0,float(reg_model.predict(X)[0])); prob=float(fault_model.predict_proba(X)[0,1]); flag=int(prob>=.5); _runtime['prediction_requests']+=1
    return {'station_id':item.station_id,'next_hour_demand_kwh':round(demand,3),'fault_probability':round(prob,4),'station_fault_flag':flag,'fault_label':'SERVER_TERMINATION_RISK' if flag else 'NORMAL','classification_target_note':model_meta.get('classification_target_note'),'station_segment':int(station_segments.get(item.station_id,-1)),'model_version':VERSION}

@app.get('/')
def dashboard(): return FileResponse(STATIC/'index.html')
@app.get('/health')
def health(): return {'status':'ok','service':'ChargeUp','version':VERSION,'dataset':'City of Palo Alto ChargePoint CY20Q4','models_loaded':bool(reg_model is not None and fault_model is not None),'timestamp_utc':datetime.now(timezone.utc).isoformat()}
@app.get('/metrics')
def metrics(): return {'runtime':_runtime,'model_meta':model_meta}
@app.post('/predict')
def predict(item:ChargeUpInput): return predict_one(item)
@app.post('/predict/demand')
def predict_demand(item:ChargeUpInput):
    r=predict_one(item); return {'station_id':r['station_id'],'next_hour_demand_kwh':r['next_hour_demand_kwh'],'station_segment':r['station_segment']}
@app.post('/predict/fault')
def predict_fault(item:ChargeUpInput):
    r=predict_one(item); return {'station_id':r['station_id'],'fault_probability':r['fault_probability'],'station_fault_flag':r['station_fault_flag'],'fault_label':r['fault_label'],'classification_target_note':r['classification_target_note']}
@app.post('/drift/check')
def drift_check(items:List[ChargeUpInput]):
    if not items: raise HTTPException(status_code=400,detail='Provide at least one observation.')
    X=build_model_features(pd.DataFrame([i.model_dump() for i in items]),station_segments); _runtime['drift_checks']+=1; return check_drift(X,baseline_stats)
@app.post('/retrain/trigger')
def retrain_trigger():
    flag=MODELS/'retrain_requested.flag'; flag.write_text(datetime.now(timezone.utc).isoformat(),encoding='utf-8'); _runtime['retrain_requests']+=1; return {'status':'queued','message':'Retraining request recorded.','next_step':'Run the Palo Alto SLM pipeline to rebuild v2 artifacts.'}
@app.get('/demo/stations')
def demo_stations(limit:int=8):
    if not DATA.exists(): return []
    df=pd.read_csv(DATA,parse_dates=['timestamp']); latest=df.sort_values('timestamp').groupby('station_id',as_index=False).tail(1).head(max(1,min(limit,20)))
    out=[]
    for _,r in latest.iterrows():
        item=ChargeUpInput(station_id=str(r.station_id),timestamp=pd.Timestamp(r.timestamp).to_pydatetime(),latitude=float(r.latitude),longitude=float(r.longitude),session_count=int(r.session_count),avg_session_minutes=float(r.avg_session_minutes),avg_total_duration_minutes=float(r.avg_total_duration_minutes),avg_power_kw=float(r.avg_power_kw),energy_kwh=float(r.energy_kwh),fee_usd=float(r.fee_usd),level1_share=float(r.level1_share),level2_share=float(r.level2_share),dc_fast_share=float(r.dc_fast_share)); out.append(predict_one(item))
    return out
