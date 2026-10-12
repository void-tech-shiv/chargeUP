"""Run the canonical ChargeUp SLM pipeline on the real Palo Alto dataset."""
from __future__ import annotations
import argparse,subprocess,sys

MODULES=[
    ['src.week03_linear_regression'],
    ['src.week04_regularized_models'],
    ['src.week05_cv_selection'],
    ['src.week06_decision_tree'],
    ['src.week07_random_forest'],
    ['src.week08_xgboost_shap'],
    ['src.week09_station_clustering'],
    ['src.week10_nested_validation'],
    ['src.week11_package_champion'],
]

def run(cmd):
    print('\n>>>',' '.join(cmd),flush=True); subprocess.run(cmd,check=True)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--export-only',action='store_true',help='use the smaller real 10k portal export mirror for a faster demo'); p.add_argument('--no-mlflow',action='store_true'); p.add_argument('--skip-download',action='store_true'); args=p.parse_args()
    prep=[sys.executable,'-m','src.palo_alto_data']
    if not args.skip_download: prep.append('--download')
    if args.export_only: prep.append('--export-only')
    run(prep)
    for module in MODULES:
        cmd=[sys.executable,'-m',module[0]]
        if module[0]=='src.week03_linear_regression' and args.no_mlflow: cmd.append('--no-mlflow')
        run(cmd)
    run([sys.executable,'-m','pytest','-q'])
    print('\nChargeUp Palo Alto real-data pipeline completed successfully.')
if __name__=='__main__': main()
