@echo off
setlocal
cd /d "%~dp0\..\.."
python -m pip install -U kaggle pyyaml
python -m kaggle --help > nul
echo.
echo Done. Next:
echo   1. Put kaggle.json files into .kaggle_tokens\account1, account2, account3
echo   2. Edit configs\kaggle_accounts.yaml
echo   3. Run scripts\windows\01_prepare_jobs.bat
