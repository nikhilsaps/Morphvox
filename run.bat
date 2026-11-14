@echo off
REM VoiceChanger - Flet GUI in the aio310 (Python 3.10) venv
setlocal
cd /d "%~dp0"
set "PY=%USERPROFILE%\.pyvenvs\aio310\Scripts\python.exe"
if not exist "%PY%" ( echo [ERROR] venv aio310 not found. & pause & exit /b 1 )
"%PY%" flet_app.py
if errorlevel 1 pause
