#!/usr/bin/env bash

echo "======================================================="
echo "   AI OCR SYSTEM - AUTOMATED SETUP (LINUX / MAC)"
echo "======================================================="
echo ""

# 1. Check Python
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] python3 could not be found. Please install Python 3.10 or 3.11."
    exit 1
fi
echo "[OK] Python is installed: $(python3 --version)"

# 2. Check Node.js
if ! command -v node &> /dev/null; then
    echo "[ERROR] node could not be found. Please install Node.js (LTS version)."
    exit 1
fi
echo "[OK] Node.js is installed: $(node --version)"

# 3. Setup Backend
echo "-------------------------------------------------------"
echo "[1/3] Setting up Backend Python Virtual Environment..."
echo "-------------------------------------------------------"
cd ai-ocr-backend

if [ ! -d "venv" ]; then
    echo "Creating virtual environment 'venv'..."
    python3 -m venv venv
fi

if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        cp .env.example .env
    fi
fi

mkdir -p uploads processed results

source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo "[OK] Backend dependencies installed!"
echo ""

# 4. Setup Frontend
echo "-------------------------------------------------------"
echo "[2/3] Setting up Frontend (Next.js & React)..."
echo "-------------------------------------------------------"
cd ../ai-ocr-frontend
npm install

echo "[OK] Frontend dependencies installed!"
echo ""

# 5. Check Ollama
echo "-------------------------------------------------------"
echo "[3/3] Checking Ollama AI Service..."
echo "-------------------------------------------------------"
cd ..
if command -v ollama &> /dev/null; then
    echo "[OK] Ollama found. Pulling qwen2.5:1.5b..."
    ollama pull qwen2.5:1.5b
else
    echo "[NOTICE] Ollama not found. The system will use offline SymSpell fallback."
fi

echo ""
echo "======================================================="
echo "   SETUP COMPLETE! Run './run.sh' to start."
echo "======================================================="
