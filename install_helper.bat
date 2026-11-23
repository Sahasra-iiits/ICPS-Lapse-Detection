@echo off
rem Double-click this once (Windows). It prepares Python if needed and connects
rem the ICPS Focus Alerts extension to the local detector.
cd /d "%~dp0"
echo ICPS Focus Alerts - one-time setup
if not exist .venv\Scripts\python.exe (
  echo Creating the Python environment ^(first time only, a few minutes^)...
  python -m venv .venv
  .venv\Scripts\python -m pip install -q --upgrade pip
  .venv\Scripts\pip install -q -r requirements.txt
)
.venv\Scripts\python install_native_host.py %*
pause
