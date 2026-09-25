@echo off
setlocal
cd /d "%~dp0"
rem BladeBot launcher for Windows - PRACTICE USE ONLY (see README.md)
set "PY=python"
where py >nul 2>nul && set "PY=py -3"
if not exist ".venv\Scripts\python.exe" (
  echo Creating a virtual environment and installing requirements...
  %PY% -m venv .venv || goto :error
  ".venv\Scripts\python.exe" -m pip install --upgrade pip
  ".venv\Scripts\python.exe" -m pip install -r requirements.txt || goto :error
)
".venv\Scripts\python.exe" -m bladebot %*
if errorlevel 1 pause
goto :eof
:error
echo.
echo Setup failed. Install Python 3.9 or newer from https://www.python.org/downloads/
echo (tick "Add python.exe to PATH" in the installer) and run start.bat again.
pause
