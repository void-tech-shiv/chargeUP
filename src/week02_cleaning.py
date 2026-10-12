"""SLM Week 2 compatibility entry point for the real Palo Alto pipeline.

Canonical implementation lives in `src.palo_alto_data`; this module keeps the
week-numbered command stable while avoiding the obsolete synthetic schema.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
from .palo_alto_data import RAW_PATH,PROCESSED_PATH,REPORT_PATH,SAMPLE_PATH,prepare


def main()->None:
    p=argparse.ArgumentParser(description='ChargeUp Week 2 Palo Alto data preparation')
    p.add_argument('--source',type=Path,default=RAW_PATH)
    p.add_argument('--output',type=Path,default=PROCESSED_PATH)
    p.add_argument('--report',type=Path,default=REPORT_PATH)
    p.add_argument('--sample',type=Path,default=SAMPLE_PATH)
    p.add_argument('--download',action='store_true')
    p.add_argument('--export-only',action='store_true')
    args=p.parse_args()
    report=prepare(source_path=args.source,processed_path=args.output,report_path=args.report,sample_path=args.sample,download=args.download or not args.source.exists(),export_only=args.export_only)
    print('ChargeUp — SLM Week 2 (Palo Alto real data)')
    print(f"Source sessions: {report['raw_rows']:,}")
    print(f"Station-hour rows: {report['station_hour_rows']:,}")
    print(f"Stations: {report['station_count']:,}")
    print(f"Classification positive rate: {report['classification_target_positive_rate']:.4%}")
    print(json.dumps({'output':str(args.output),'report':str(args.report)},indent=2))

if __name__=='__main__': main()
