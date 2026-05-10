@echo off
REM NetPulse Setup and Run Script for Windows (Batch)
REM This script sets up the environment and runs NetPulse

echo ==========================================
echo   NetPulse - Network Anomaly Detector
echo   Setup and Run Script (Windows)
echo ==========================================
echo.

REM Get the directory where this script is located
set "SCRIPT_DIR=%~dp0"
set "PROJECT_DIR=%SCRIPT_DIR%.."

cd /d "%PROJECT_DIR%"

echo [1/5] Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH. Please install Python 3.8 or higher.
    pause
    exit /b 1
)
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo   ^> Python %PYTHON_VERSION% found

echo.
echo [2/5] Creating virtual environment...
if not exist "venv" (
    python -m venv venv
    echo   ^> Virtual environment created
) else (
    echo   ^> Virtual environment already exists
)

echo.
echo [3/5] Activating virtual environment and installing dependencies...
call venv\Scripts\activate.bat
python -m pip install --upgrade pip >nul 2>&1
pip install -e . >nul 2>&1
echo   ^> Dependencies installed

echo.
echo [4/5] Creating necessary directories...
if not exist "%USERPROFILE%\.netpulse\models\current" mkdir "%USERPROFILE%\.netpulse\models\current"
if not exist "%USERPROFILE%\.netpulse\data" mkdir "%USERPROFILE%\.netpulse\data"
if not exist "%USERPROFILE%\.netpulse\reports" mkdir "%USERPROFILE%\.netpulse\reports"
echo   ^> Directories created

echo.
echo [5/5] NetPulse is ready!
echo.
echo ==========================================
echo   Usage Instructions:
echo ==========================================
echo.
echo   Start monitoring (requires Administrator):
echo     scripts\run.bat start
echo.
echo   Start monitoring on specific interface:
echo     scripts\run.bat start --interface Ethernet
echo.
echo   View live status dashboard:
echo     scripts\run.bat status
echo.
echo   Generate reports:
echo     scripts\run.bat report --last 24h
echo     scripts\run.bat report --last 7d --export alerts.csv
echo.
echo   Train models manually:
echo     scripts\run.bat train
echo.
echo   Run evaluation:
echo     scripts\run.bat eval
echo.
echo   Edit configuration:
echo     scripts\run.bat config
echo.
echo   Reset baseline:
echo     scripts\run.bat reset-baseline
echo.
echo ==========================================

REM If a command was passed as argument, execute it
if "%~1"=="" (
    echo No command specified. Showing help...
    echo.
    python -m netpulse --help
) else (
    echo.
    echo Executing: netpulse %*
    echo.
    python -m netpulse %*
)

pause
