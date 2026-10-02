#!/bin/bash
set -e

echo "Starting Advanced AI Counsellor..."

# Start FastAPI internally
python -m uvicorn backend.main:app \
    --host 0.0.0.0 \
    --port 8000 &

BACKEND_PID=$!

echo "FastAPI started with PID $BACKEND_PID"

# Give FastAPI a few seconds to initialize
sleep 5

# Start Streamlit on the Hugging Face exposed port
python -m streamlit run frontend/app.py \
    --server.address 0.0.0.0 \
    --server.port 7860 \
    --server.headless true

# Stop backend if Streamlit exits
kill $BACKEND_PID