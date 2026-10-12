#!/usr/bin/env sh
set -eu
[ -f data/raw/chargeup_sessions.csv ] || python -m src.generate_data
python -m src.week02_cleaning
python -m src.week11_package_champion
exec uvicorn api.main:app --host 0.0.0.0 --port "${PORT:-8000}"
