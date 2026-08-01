@echo off
setlocal
cd /d "%~dp0\..\.."
python scripts\kaggle_orchestrator.py --config configs\kaggle_accounts.yaml --action all

