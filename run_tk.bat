@echo off
REM Fallback Tkinter GUI if Flet misbehaves
setlocal
cd /d "%~dp0"
set "PY=%USERPROFILE%\.pyvenvs\aio310\Scripts\python.exe"
"%PY%" gui.py
if errorlevel 1 pause
