#!/bin/bash
set -e

echo "Starting Advanced AI Counsellor..."

# Start FastAPI internally on port 8000
python -m uvicorn backend.main:app \
    --host 0.0.0.0 \
    --port 8000 &

BACKEND_PID=$!
echo "FastAPI started with PID $BACKEND_PID"

# Give FastAPI a few seconds to initialize
sleep 5

# Start Streamlit on the Hugging Face exposed port 7860
python -m streamlit run frontend/app.py \
    --server.address 0.0.0.0 \
    --server.port 7860 \
    --server.headless true &

STREAMLIT_PID=$!

# Wait for either process to exit
wait -n $BACKEND_PID $STREAMLIT_PID

# Kill both if one exits
kill $BACKEND_PID $STREAMLIT_PID 2>/dev/null || true