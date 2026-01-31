@echo off
REM ====================================================================
REM Autonomous AI Agent - Build Script for Windows
REM ====================================================================

echo ========================================================
echo   Autonomous AI Agent - Windows Build Script
echo ========================================================
echo.

REM Get the directory where this script is located
set SCRIPT_DIR=%~dp0
set PROJECT_DIR=%SCRIPT_DIR%

echo Project Directory: %PROJECT_DIR%
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher from https://python.org
    pause
    exit /b 1
)

echo Python found. Installing dependencies...
echo.

REM Install dependencies
echo Installing required packages...
pip install -r "%PROJECT_DIR%requirements.txt"
if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)

REM Install PyInstaller if not already installed
pip show pyinstaller >nul 2>&1
if errorlevel 1 (
    echo Installing PyInstaller...
    pip install pyinstaller
    if errorlevel 1 (
        echo ERROR: Failed to install PyInstaller
        pause
        exit /b 1
    )
)

echo.
echo ========================================================
echo Building Windows Executable...
echo ========================================================
echo.

REM Clean previous build artifacts
if exist "%PROJECT_DIR%dist" rmdir /s /q "%PROJECT_DIR%dist"
if exist "%PROJECT_DIR%build" rmdir /s /q "%PROJECT_DIR%build"

REM Build the executable
echo Running PyInstaller...
pyinstaller "%PROJECT_DIR%autonomous_agent.spec" --noconfirm

if exist "%PROJECT_DIR%dist\AutonomousAI-Agent" (
    echo.
    echo ========================================================
    echo BUILD SUCCESSFUL!
    echo ========================================================
    echo.
    echo Executable location: %PROJECT_DIR%dist\AutonomousAI-Agent\AutonomousAI-Agent.exe
    echo.
    echo Next steps:
    echo 1. Copy the entire "dist\AutonomousAI-Agent" folder
    echo 2. Distribute to other Windows machines
    echo 3. Users should configure their API keys via Settings
    echo.
    echo API Key Setup:
    echo - Gemini: https://aistudio.google.com/
    echo - OpenAI: https://platform.openai.com/api-keys
    echo.
) else (
    echo.
    echo ========================================================
    echo BUILD FAILED
    echo ========================================================
    echo.
    echo Please check the error messages above.
    echo Common issues:
    echo - Missing Python dependencies
    echo - Antivirus software blocking the build
    echo - Insufficient disk space
    echo.
)

pause
