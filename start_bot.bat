@echo off
echo Starting Trading Bot...
cd /d "%~dp0"
call .venv\Scripts\activate.bat
python run_realtime.py --live
pause
