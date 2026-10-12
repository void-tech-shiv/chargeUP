$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
uvicorn api.main:app --reload
