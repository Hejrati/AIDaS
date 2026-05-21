@echo off
cd /d "%~dp0"
conda run -n oct-segmenter-env python app.py
pause
