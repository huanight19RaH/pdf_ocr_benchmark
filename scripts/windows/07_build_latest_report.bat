@echo off
setlocal
cd /d "%~dp0\..\.."
python scripts\build_latest_report.py --input-dir kaggle_remote_jobs\outputs --output-dir outputs\final_benchmark_report_latest

