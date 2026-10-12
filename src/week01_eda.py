"""SLM Week 1: initial EDA for the real City of Palo Alto ChargePoint data."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import pandas as pd
from .palo_alto_data import PROCESSED_PATH,REPORT_PATH,prepare
from .palo_alto_schema import REGRESSION_TARGET,CLASSIFICATION_TARGET,FAULT_PROXY_DESCRIPTION


def ensure_data(download:bool=False)->None:
    if download or not PROCESSED_PATH.exists():
        prepare(download=True)


def run_eda(data_path:Path=PROCESSED_PATH)->None:
    df=pd.read_csv(data_path,parse_dates=['timestamp'])
    print('\n=== ChargeUp Week 1 — Palo Alto real-data EDA ===')
    print(f'Rows (station-hour): {len(df):,}')
    print(f'Stations: {df.station_id.nunique():,}')
    print(f'Time range: {df.timestamp.min()} to {df.timestamp.max()}')
    print(f'Regression target: {REGRESSION_TARGET}')
    print(f'Classification target: {CLASSIFICATION_TARGET}')
    print(f'Classification note: {FAULT_PROXY_DESCRIPTION}')
    print('\nNext-hour demand summary:')
    print(df[REGRESSION_TARGET].describe().to_string())
    print(f'\nServer-termination proxy positive rate: {df[CLASSIFICATION_TARGET].mean():.4%}')
    print('\nTop hours by mean next-hour demand:')
    hourly=df.assign(hour=df.timestamp.dt.hour).groupby('hour')[REGRESSION_TARGET].mean().sort_values(ascending=False)
    print(hourly.head(8).to_string())
    print('\nTop stations by observed energy:')
    station=df.groupby('station_id')['energy_kwh'].agg(['count','sum','mean']).sort_values('sum',ascending=False)
    print(station.head(10).to_string())
    print('\nGeography:')
    print(f"Latitude: {df.latitude.min():.6f} to {df.latitude.max():.6f}")
    print(f"Longitude: {df.longitude.min():.6f} to {df.longitude.max():.6f}")
    if REPORT_PATH.exists():
        report=json.loads(REPORT_PATH.read_text(encoding='utf-8'))
        print(f"\nSource sessions: {report.get('raw_rows',0):,}")
        print(f"Server-ended source sessions: {report.get('server_terminated_sessions',0):,}")
    print('\nWeather/voltage/current are not present in the supplied City CSV and are not fabricated.')


def main()->None:
    p=argparse.ArgumentParser(description='ChargeUp Week 1 Palo Alto EDA')
    p.add_argument('--data',type=Path,default=PROCESSED_PATH)
    p.add_argument('--download',action='store_true')
    args=p.parse_args(); ensure_data(args.download); run_eda(args.data)

if __name__=='__main__': main()
