@echo off
cd /d "%~dp0"
conda run -n aidas-env python app.py
pause
