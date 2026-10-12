from fastapi.testclient import TestClient
from api.main import app

client=TestClient(app)

def test_health():
    r=client.get('/health'); assert r.status_code==200; body=r.json(); assert body['status']=='ok'; assert body['version']=='chargeup-paloalto-v2.0.0'

def test_predict():
    payload={'station_id':'PALO ALTO CA / HAMILTON #1','timestamp':'2020-12-01T18:00:00','latitude':37.444572,'longitude':-122.160309,'session_count':2,'avg_session_minutes':75.0,'avg_total_duration_minutes':110.0,'avg_power_kw':6.1,'energy_kwh':14.2,'fee_usd':0.0,'level1_share':0.0,'level2_share':1.0,'dc_fast_share':0.0}
    r=client.post('/predict',json=payload); assert r.status_code==200; body=r.json(); assert 'next_hour_demand_kwh' in body; assert 'fault_probability' in body; assert body['model_version']=='chargeup-paloalto-v2.0.0'
