#!/usr/bin/env bash

echo "======================================================="
echo "        LAUNCHING AI OCR SYSTEM (FASTAPI + NEXT.JS)"
echo "======================================================="
echo ""

# Start backend
cd ai-ocr-backend
source venv/bin/activate
python3 -m uvicorn app:app --reload --host 127.0.0.1 --port 8000 &
BACKEND_PID=$!

# Start frontend
cd ../ai-ocr-frontend
npm run dev &
FRONTEND_PID=$!

echo "Backend PID: $BACKEND_PID"
echo "Frontend PID: $FRONTEND_PID"
echo "Opening http://localhost:3000..."

sleep 3
if command -v xdg-open &> /dev/null; then
    xdg-open http://localhost:3000
elif command -v open &> /dev/null; then
    open http://localhost:3000
fi

wait $BACKEND_PID $FRONTEND_PID
