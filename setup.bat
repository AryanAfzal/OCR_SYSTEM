@echo off
setlocal EnableDelayedExpansion

echo =======================================================
echo    AI OCR SYSTEM - BULLETPROOF ONE-CLICK SETUP
echo =======================================================
echo.

:: 1. Detect Python (Prefer Python 3.11 or 3.10 for PyTorch and PaddlePaddle)
set "PY_CMD="

py -3.11 --version >nul 2>&1
if %errorlevel% equ 0 (
    set "PY_CMD=py -3.11"
    goto :PYTHON_FOUND
)

py -3.10 --version >nul 2>&1
if %errorlevel% equ 0 (
    set "PY_CMD=py -3.10"
    goto :PYTHON_FOUND
)

python3.11 --version >nul 2>&1
if %errorlevel% equ 0 (
    set "PY_CMD=python3.11"
    goto :PYTHON_FOUND
)

python --version >nul 2>&1
if %errorlevel% equ 0 (
    set "PY_CMD=python"
    goto :PYTHON_FOUND
)

echo [ERROR] No compatible Python installation found!
echo Please install Python 3.11 or 3.10 from https://www.python.org/
echo Note: Remember to check the box: Add Python to PATH during installation.
echo.
pause
exit /b 1

:PYTHON_FOUND
echo [OK] Using Python command: %PY_CMD%
%PY_CMD% --version
echo.

:: 2. Detect Node.js
node --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Node.js is not installed or not in PATH!
    echo Please install Node.js LTS from https://nodejs.org/
    echo.
    pause
    exit /b 1
)
echo [OK] Node.js detected:
node --version
echo.

:: 3. Setup Backend Environment
echo -------------------------------------------------------
echo [1/3] Setting up Backend Virtual Environment...
echo -------------------------------------------------------
cd ai-ocr-backend

if not exist "venv\Scripts\python.exe" (
    echo Creating clean virtual environment with %PY_CMD%...
    %PY_CMD% -m venv venv
)

if not exist ".env" (
    if exist ".env.example" (
        echo Creating .env configuration file...
        copy .env.example .env >nul
    )
)

if not exist "uploads" mkdir uploads
if not exist "processed" mkdir processed
if not exist "results" mkdir results

echo Upgrading pip and installing strictly pinned libraries...
echo Please wait, this takes 2-4 minutes on first run...
call .\venv\Scripts\activate.bat
.\venv\Scripts\python.exe -m pip install --upgrade pip
.\venv\Scripts\python.exe -m pip install -r requirements.txt

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Dependency installation failed!
    echo Please ensure you are connected to the internet and run setup.bat again.
    pause
    exit /b 1
)

echo.
echo [OK] Backend packages installed successfully!
echo.

:: Check TrOCR local weights folder
if exist "trocr-base-handwritten" (
    echo [OK] Local TrOCR offline model folder found.
) else (
    echo [NOTICE] trocr-base-handwritten local folder not found.
    echo TrOCR will auto-download from Hugging Face on first run.
)
echo.

:: 4. Setup Frontend
echo -------------------------------------------------------
echo [2/3] Setting up Frontend Next.js and React...
echo -------------------------------------------------------
cd ..\ai-ocr-frontend

echo Installing npm dependencies...
call npm install

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Frontend npm install failed!
    pause
    exit /b 1
)

echo [OK] Frontend packages installed successfully!
echo.

:: 5. Check Ollama
echo -------------------------------------------------------
echo [3/3] Checking Ollama AI Service...
echo -------------------------------------------------------
cd ..
ollama --version >nul 2>&1
if %errorlevel% equ 0 (
    echo [OK] Ollama is installed:
    ollama --version
    echo Pulling recommended AI model qwen2.5:1.5b...
    ollama pull qwen2.5:1.5b
) else (
    echo [NOTICE] Ollama not found in PATH.
    echo The system will use fast offline SymSpell spell checking by default.
    echo To enable LLM post-correction, install Ollama from https://ollama.ai
)

echo.
echo =======================================================
echo    SETUP COMPLETE! YOU ARE READY TO GO!
echo =======================================================
echo To start the project, simply double-click:
echo                     run.bat
echo =======================================================
echo.
pause
