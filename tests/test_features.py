import pandas as pd
from src.features import build_features,MODEL_FEATURES

def test_feature_shape():
    df=pd.DataFrame([{
        'station_id':'PALO ALTO CA / HAMILTON #1','timestamp':'2020-12-01T18:00:00','latitude':37.444572,'longitude':-122.160309,
        'session_count':2,'avg_session_minutes':75.0,'avg_total_duration_minutes':110.0,'avg_power_kw':6.1,'energy_kwh':14.2,'fee_usd':0.0,
        'level1_share':0.0,'level2_share':1.0,'dc_fast_share':0.0,
    }])
    X=build_features(df,{'PALO ALTO CA / HAMILTON #1':2})
    assert list(X.columns)==MODEL_FEATURES
    assert X.shape==(1,len(MODEL_FEATURES))
    assert int(X['station_segment'].iloc[0])==2
