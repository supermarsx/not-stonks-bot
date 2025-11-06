@echo off
REM Day Trading Orchestrator - Windows Startup Script

setlocal enabledelayedexpansion

REM Colors for Windows CMD
set "BLUE=[94m"
set "GREEN=[92m"
set "YELLOW=[93m"
set "RED=[91m"
set "NC=[0m"

REM Banner
echo.
echo %BLUE%╔══════════════════════════════════════════════════════════════╗%
echo %BLUE%║                                                              ║%
echo %BLUE%║           DAY TRADING ORCHESTRATOR STARTUP                   ║%
echo %BLUE%║                    Windows Script                            ║%
echo %BLUE%║                                                              ║%
echo %BLUE%╚══════════════════════════════════════════════════════════════╝%
echo %NC%

REM Get script directory
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

echo %BLUE%[INFO]%NC% Starting Day Trading Orchestrator...
echo %BLUE%[INFO]%NC% Script directory: %SCRIPT_DIR%

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo %RED%[ERROR]%NC% Python is not installed or not in PATH
    echo Please install Python 3.8+ and add it to PATH
    pause
    exit /b 1
)

echo %GREEN%[SUCCESS]%NC% Python found
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo %GREEN%[SUCCESS]%NC% Python version: %PYTHON_VERSION%

REM Check if virtual environment exists
if not exist "venv" (
    echo %YELLOW%[WARNING]%NC% Virtual environment not found
    echo %BLUE%[INFO]%NC% Creating virtual environment...
    
    python -m venv venv
    if errorlevel 1 (
        echo %RED%[ERROR]%NC% Failed to create virtual environment
        pause
        exit /b 1
    )
    echo %GREEN%[SUCCESS]%NC% Virtual environment created
)

REM Activate virtual environment
echo %BLUE%[INFO]%NC% Activating virtual environment...
call venv\Scripts\activate.bat

REM Check if requirements are installed
if not exist "requirements_installed" (
    echo %BLUE%[INFO]%NC% Installing dependencies...
    pip install -r trading_orchestrator\requirements.txt
    if errorlevel 1 (
        echo %RED%[ERROR]%NC% Failed to install dependencies
        pause
        exit /b 1
    )
    type nul > requirements_installed
    echo %GREEN%[SUCCESS]%NC% Dependencies installed
)

REM Create necessary directories
echo %BLUE%[INFO]%NC% Creating necessary directories...
if not exist logs mkdir logs
if not exist data mkdir data
if not exist backups mkdir backups

REM Set environment variables
set PYTHONPATH=%SCRIPT_DIR%;%PYTHONPATH%

REM Start the application
echo %BLUE%[INFO]%NC% Launching Day Trading Orchestrator...
echo.

REM Handle different startup modes
if "%1"=="demo" (
    echo %BLUE%[INFO]%NC% Starting in DEMO mode...
    python main.py --demo
) else if "%1"=="debug" (
    echo %BLUE%[INFO]%NC% Starting in DEBUG mode...
    python main.py --debug
) else if "%1"=="create-config" (
    echo %BLUE%[INFO]%NC% Creating default configuration...
    python main.py --create-config
    echo %YELLOW%[WARNING]%NC% Please edit config.json with your API keys before starting!
) else if "%1"=="health-check" (
    echo %BLUE%[INFO]%NC% Running health check...
    python health_check.py
) else (
    echo %BLUE%[INFO]%NC% Starting in NORMAL mode...
    python main.py
)

if errorlevel 1 (
    echo %RED%[ERROR]%NC% Application encountered an error
    pause
    exit /b 1
)

echo %GREEN%[SUCCESS]%NC% Application shutdown complete
pause