FROM python:3.12-slim

WORKDIR /app

# System dependencies needed by scipy / arch
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

EXPOSE 8050

# Render injects $PORT at runtime; fall back to 8050 locally
CMD gunicorn --bind "0.0.0.0:${PORT:-8050}" \
             --timeout 600 \
             --workers 1 \
             --log-level info \
             main:server
