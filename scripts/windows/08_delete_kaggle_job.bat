@echo off
setlocal
cd /d "%~dp0\..\.."
if "%~1"=="" (
  echo Usage: scripts\windows\08_delete_kaggle_job.bat JOB_NAME
  echo Example: scripts\windows\08_delete_kaggle_job.bat paddleocr-vl
  exit /b 1
)
python scripts\kaggle_orchestrator.py --config configs\kaggle_accounts.yaml --action delete --job %1

