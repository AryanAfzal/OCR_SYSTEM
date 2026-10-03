@echo off
setlocal enabledelayedexpansion

echo =======================================================
echo        LAUNCHING AI OCR SYSTEM (FASTAPI + NEXT.JS)
echo =======================================================
echo.

:: 1. Verify Backend venv exists
if not exist "ai-ocr-backend\venv" (
    echo [ERROR] Virtual environment not found!
    echo Please run setup.bat first to install all dependencies.
    echo.
    pause
    exit /b 1
)

:: 2. Launch Ollama in background if available
ollama --version >nul 2>&1
if %errorlevel% equ 0 (
    echo Starting Ollama AI Service in background...
    start "Ollama AI" /min cmd /c "ollama run qwen2.5:1.5b"
)

:: 3. Launch Backend in a new window
echo Starting Backend (FastAPI on http://127.0.0.1:8000)...
start "AI OCR Backend (FastAPI)" cmd /k "cd ai-ocr-backend && call .\venv\Scripts\activate.bat && python -m uvicorn app:app --reload --host 127.0.0.1 --port 8000"

:: 4. Launch Frontend in a new window
echo Starting Frontend (Next.js on http://localhost:3000)...
start "AI OCR Frontend (Next.js)" cmd /k "cd ai-ocr-frontend && npm run dev"

:: 5. Wait 3 seconds and open browser
echo Waiting for servers to initialize...
timeout /t 4 /nobreak >nul
start http://localhost:3000

echo.
echo =======================================================
echo  Application is running!
echo  - Frontend: http://localhost:3000
echo  - Backend:  http://127.0.0.1:8000
echo  (To stop, simply close the opened terminal windows)
echo =======================================================
echo.
