#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
uvicorn api.main:app --reload
