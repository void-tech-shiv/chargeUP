from fastapi.testclient import TestClient
from api.main import app

client=TestClient(app)

def test_health():
    r=client.get('/health'); assert r.status_code==200; body=r.json(); assert body['status']=='ok'; assert body['version']=='chargeup-paloalto-v2.0.0'

def test_predict():
    payload={'station_id':'PALO ALTO CA / HAMILTON #1','timestamp':'2020-12-01T18:00:00','latitude':37.444572,'longitude':-122.160309,'session_count':2,'avg_session_minutes':75.0,'avg_total_duration_minutes':110.0,'avg_power_kw':6.1,'energy_kwh':14.2,'fee_usd':0.0,'level1_share':0.0,'level2_share':1.0,'dc_fast_share':0.0}
    r=client.post('/predict',json=payload); assert r.status_code==200; body=r.json(); assert 'next_hour_demand_kwh' in body; assert 'fault_probability' in body; assert body['model_version']=='chargeup-paloalto-v2.0.0'; assert body['prediction_timestamp'].startswith('2020-12-01T18:00:00')

def test_feature_importance_insight():
    r=client.get('/insights/feature-importance'); assert r.status_code==200; body=r.json(); assert body['regression_shap']; assert body['classification_rf_permutation']; assert 'feature' in body['regression_shap'][0]

def test_cluster_insight():
    r=client.get('/insights/clusters'); assert r.status_code==200; body=r.json(); assert body['stations']>0; assert body['points']; assert {'station_id','kmeans_segment','pca_1','pca_2'}.issubset(body['points'][0])

def test_real_data_drift_demo():
    r=client.post('/drift/demo?limit=25'); assert r.status_code==200; body=r.json(); assert body['observations']==25; assert body['status'] in {'STABLE','MODERATE_DRIFT','HIGH_DRIFT'}; assert 'feature_psi' in body
