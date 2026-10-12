import pandas as pd
from src.palo_alto_data import clean_sessions,aggregate_station_hour

def test_exact_next_hour_targets_and_server_proxy():
    raw=pd.DataFrame([
        {'Station Name':'A','Start Date':'1/1/2020 10:10','End Date':'1/1/2020 11:00','Total Duration (hh:mm:ss)':'0:50:00','Charging Time (hh:mm:ss)':'0:45:00','Energy (kWh)':5.0,'Port Type':'Level 2','Port Number':1,'Latitude':37.4,'Longitude':-122.1,'Fee':0,'Ended By':'Customer'},
        {'Station Name':'A','Start Date':'1/1/2020 11:05','End Date':'1/1/2020 12:00','Total Duration (hh:mm:ss)':'0:55:00','Charging Time (hh:mm:ss)':'0:50:00','Energy (kWh)':7.0,'Port Type':'Level 2','Port Number':1,'Latitude':37.4,'Longitude':-122.1,'Fee':0,'Ended By':'CPS Server'},
        {'Station Name':'A','Start Date':'1/1/2020 13:05','End Date':'1/1/2020 14:00','Total Duration (hh:mm:ss)':'0:55:00','Charging Time (hh:mm:ss)':'0:50:00','Energy (kWh)':9.0,'Port Type':'Level 2','Port Number':1,'Latitude':37.4,'Longitude':-122.1,'Fee':0,'Ended By':'Customer'},
    ])
    sessions,_=clean_sessions(raw); hourly=aggregate_station_hour(sessions).set_index('timestamp')
    row10=hourly.loc[pd.Timestamp('2020-01-01 10:00:00')]
    row11=hourly.loc[pd.Timestamp('2020-01-01 11:00:00')]
    assert row10['next_hour_demand_kwh']==7.0
    assert int(row10['station_fault_flag'])==1
    assert row11['next_hour_demand_kwh']==0.0
    assert int(row11['station_fault_flag'])==0
