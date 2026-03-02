#!/bin/bash

# Start Dashboard Script
# Starts both backend and frontend servers

echo "=============================================="
echo "CHG Framework Dashboard Startup"
echo "=============================================="

cd "$(dirname "$0")/.."

# Check if API keys are set
if [ -z "$OPENAI_API_KEY" ] || [ -z "$GOOGLE_API_KEY" ]; then
    echo ""
    echo "⚠️  Warning: API keys not set"
    echo "   Set OPENAI_API_KEY and GOOGLE_API_KEY for full functionality"
    echo ""
fi

# Start backend
echo "Starting backend server..."
cd "$(dirname "$0")/.."
./venv/bin/python -m uvicorn src.web.main:app --reload --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

echo "Backend started on http://localhost:8000 (PID: $BACKEND_PID)"
echo "API docs available at http://localhost:8000/docs"

# Wait a bit for backend to start
sleep 3

# Start frontend
echo ""
echo "Starting frontend server..."
cd frontend

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "Installing frontend dependencies..."
    npm install
fi

npm run dev &
FRONTEND_PID=$!

echo "Frontend started on http://localhost:3000 (PID: $FRONTEND_PID)"
echo ""
echo "=============================================="
echo "Dashboard is running!"
echo "=============================================="
echo ""
echo "Backend:  http://localhost:8000"
echo "Frontend: http://localhost:3000"
echo "API Docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop both servers"
echo ""

# Wait for user interrupt
trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit" INT TERM

wait

