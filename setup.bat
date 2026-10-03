@echo off
setlocal enabledelayedexpansion

echo =======================================================
echo    AI OCR SYSTEM - AUTOMATED ONE-CLICK SETUP
echo =======================================================
echo.

:: 1. Check Python Installation
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not added to PATH!
    echo Please install Python 3.10 or 3.11 from https://www.python.org/
    echo Make sure to CHECK "Add python.exe to PATH" during installation.
    echo.
    pause
    exit /b 1
)
echo [OK] Python is installed:
python --version
echo.

:: 2. Check Node.js Installation
node --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Node.js is not installed or not added to PATH!
    echo Please install Node.js (LTS version) from https://nodejs.org/
    echo.
    pause
    exit /b 1
)
echo [OK] Node.js is installed:
node --version
echo.

:: 3. Setup Backend Environment
echo -------------------------------------------------------
echo [1/3] Setting up Backend Python Virtual Environment...
echo -------------------------------------------------------
cd ai-ocr-backend

if not exist "venv" (
    echo Creating virtual environment 'venv'...
    python -m venv venv
)

if not exist ".env" (
    if exist ".env.example" (
        echo Creating .env from .env.example...
        copy .env.example .env >nul
    )
)

:: Create required runtime directories
if not exist "uploads" mkdir uploads
if not exist "processed" mkdir processed
if not exist "results" mkdir results

echo Upgrading pip and installing required backend libraries...
echo (This may take 2-4 minutes on the first run, please wait...)
call .\venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Backend dependency installation failed.
    echo Please check your internet connection and try running setup.bat again.
    pause
    exit /b 1
)

echo [OK] Backend dependencies installed successfully!
echo.

:: 4. Setup Frontend
echo -------------------------------------------------------
echo [2/3] Setting up Frontend (Next.js & React)...
echo -------------------------------------------------------
cd ..\ai-ocr-frontend

echo Installing npm dependencies...
call npm install

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Frontend npm install failed.
    pause
    exit /b 1
)

echo [OK] Frontend dependencies installed successfully!
echo.

:: 5. Check Ollama (Optional AI Post-Correction)
echo -------------------------------------------------------
echo [3/3] Checking Ollama AI Service...
echo -------------------------------------------------------
cd ..
ollama --version >nul 2>&1
if %errorlevel% equ 0 (
    echo [OK] Ollama is installed:
    ollama --version
    echo Pulling recommended AI post-correction model (qwen2.5:1.5b)...
    ollama pull qwen2.5:1.5b
) else (
    echo [NOTICE] Ollama is not detected in PATH.
    echo The system will use fast offline SymSpell spell checking by default.
    echo (Optional: For maximum LLM accuracy, download Ollama from https://ollama.ai)
)

echo.
echo =======================================================
echo    SETUP COMPLETE! YOU ARE READY TO GO!
echo =======================================================
echo To start the project at any time, simply double-click:
echo                     run.bat
echo =======================================================
echo.
pause
