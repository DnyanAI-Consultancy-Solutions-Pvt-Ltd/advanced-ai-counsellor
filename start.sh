#!/bin/bash
set -e

echo "Starting Advanced AI Counsellor..."

# Start FastAPI internally on private port 8000
python -m uvicorn backend.main:app \
    --host 127.0.0.1 \
    --port 8000 &
BACKEND_PID=$!
echo "FastAPI started with PID $BACKEND_PID"

# Give FastAPI a few seconds to initialize
sleep 3

# Start Streamlit internally on private port 8501
python -m streamlit run frontend/app.py \
    --server.address 127.0.0.1 \
    --server.port 8501 \
    --server.headless true &
STREAMLIT_PID=$!
echo "Streamlit started with PID $STREAMLIT_PID"

# Start Nginx on the Hugging Face exposed port 7860 to route traffic
echo "Starting Nginx proxy on port 7860..."
nginx -g 'daemon off;' &
NGINX_PID=$!

# Wait for any process to exit
wait -n $BACKEND_PID $STREAMLIT_PID $NGINX_PID

# Cleanup remaining processes if one crashes
kill $BACKEND_PID $STREAMLIT_PID $NGINX_PID 2>/dev/null || true