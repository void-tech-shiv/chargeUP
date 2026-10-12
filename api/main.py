from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from src.model_features import build_model_features
from src.drift import check_drift

ROOT=Path(__file__).resolve().parents[1]; MODELS=ROOT/'models'; REGISTRY=MODELS/'registry'/'chargeup-v1.0.0'; DATA=ROOT/'data/raw/chargeup_sessions.csv'; STATIC=ROOT/'api/static'
app=FastAPI(title='ChargeUp API',version='1.0.0',description='Next-hour EV charging demand forecasting + station fault alerting with drift monitoring.')
app.mount('/static',StaticFiles(directory=STATIC),name='static')
_runtime={'prediction_requests':0,'drift_checks':0,'retrain_requests':0,'started_at':datetime.now(timezone.utc).isoformat()}
reg_model=fault_model=None; station_segments={}; baseline_stats={}; model_meta={}; training_metrics={}

def load_artifacts():
    global reg_model,fault_model,station_segments,baseline_stats,model_meta,training_metrics
    required=[REGISTRY/'regression_model.joblib',REGISTRY/'fault_model.joblib',REGISTRY/'station_segments.json',REGISTRY/'baseline_stats.json',REGISTRY/'manifest.json']
    missing=[str(p) for p in required if not p.exists()]
    if missing:
        raise RuntimeError('Missing versioned artifacts. Run: python -m src.generate_data && python -m src.week02_cleaning && python -m src.week11_package_champion\n'+'\n'.join(missing))
    reg_model=joblib.load(REGISTRY/'regression_model.joblib'); fault_model=joblib.load(REGISTRY/'fault_model.joblib')
    station_segments=json.loads((REGISTRY/'station_segments.json').read_text()); baseline_stats=json.loads((REGISTRY/'baseline_stats.json').read_text()); model_meta=json.loads((REGISTRY/'manifest.json').read_text())
    metrics_path=ROOT/'reports/week10_validation_metrics.json'; training_metrics=json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
load_artifacts()

class ChargeUpInput(BaseModel):
    station_id:str='CHG-001'; timestamp:datetime=Field(default_factory=datetime.now); temperature_c:float=31.0; humidity_pct:float=64.0; rainfall_mm:float=0.0; latitude:float=16.5062; longitude:float=80.6480; session_count:int=7; avg_session_minutes:float=43.0; avg_power_kw:float=7.2; connector_errors:int=0; voltage_v:float=230.0; current_a:float=28.0; availability_pct:float=99.0; energy_kwh:float=31.5

def predict_one(item:ChargeUpInput)->dict:
    X=build_model_features(pd.DataFrame([item.model_dump()]),station_segments); demand=max(0.0,float(reg_model.predict(X)[0])); prob=float(fault_model.predict_proba(X)[0,1]); flag=int(prob>=.5); _runtime['prediction_requests']+=1
    return {'station_id':item.station_id,'next_hour_demand_kwh':round(demand,3),'fault_probability':round(prob,4),'station_fault_flag':flag,'fault_label':'FAULT' if flag else 'NORMAL','station_segment':int(station_segments.get(item.station_id,-1)),'model_version':model_meta.get('version')}

@app.get('/')
def dashboard(): return FileResponse(STATIC/'index.html')
@app.get('/health')
def health(): return {'status':'ok','service':'ChargeUp','version':model_meta.get('version'),'models_loaded':bool(reg_model is not None and fault_model is not None),'timestamp_utc':datetime.now(timezone.utc).isoformat()}
@app.get('/metrics')
def metrics(): return {'runtime':_runtime,'validation':training_metrics,'model_meta':model_meta}
@app.post('/predict')
def predict(item:ChargeUpInput): return predict_one(item)
@app.post('/predict/demand')
def predict_demand(item:ChargeUpInput):
    r=predict_one(item); return {k:r[k] for k in ['station_id','next_hour_demand_kwh','station_segment','model_version']}
@app.post('/predict/fault')
def predict_fault(item:ChargeUpInput):
    r=predict_one(item); return {k:r[k] for k in ['station_id','fault_probability','station_fault_flag','fault_label','model_version']}
@app.post('/drift/check')
def drift_check(items:List[ChargeUpInput]):
    if not items: raise HTTPException(status_code=400,detail='Provide at least one observation.')
    X=build_model_features(pd.DataFrame([i.model_dump() for i in items]),station_segments); _runtime['drift_checks']+=1; return check_drift(X,baseline_stats)
@app.post('/retrain/trigger')
def retrain_trigger():
    flag=MODELS/'retrain_requested.flag'; flag.write_text(datetime.now(timezone.utc).isoformat(),encoding='utf-8'); _runtime['retrain_requests']+=1; return {'status':'queued','message':'Retraining request recorded.','trigger':'manual','next_step':'Run python -m src.week11_package_champion after refreshed training data is prepared.'}
@app.get('/demo/stations')
def demo_stations(limit:int=8):
    if not DATA.exists(): return []
    df=pd.read_csv(DATA,parse_dates=['timestamp']); latest=df.sort_values('timestamp').groupby('station_id',as_index=False).tail(1).head(max(1,min(limit,20))); out=[]
    for _,r in latest.iterrows(): out.append(predict_one(ChargeUpInput(station_id=str(r.station_id),timestamp=pd.Timestamp(r.timestamp).to_pydatetime(),temperature_c=float(r.temperature_c),humidity_pct=float(r.humidity_pct),rainfall_mm=float(r.rainfall_mm),latitude=float(r.latitude),longitude=float(r.longitude),session_count=int(r.session_count),avg_session_minutes=float(r.avg_session_minutes),avg_power_kw=float(r.avg_power_kw),connector_errors=int(r.connector_errors),voltage_v=float(r.voltage_v),current_a=float(r.current_a),availability_pct=float(r.availability_pct),energy_kwh=float(r.energy_kwh))))
    return out
