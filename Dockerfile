FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first for better Docker caching
COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Hugging Face Docker Spaces expose port 7860
EXPOSE 7860

RUN chmod +x /app/start.sh

CMD ["/bin/bash", "/app/start.sh"]