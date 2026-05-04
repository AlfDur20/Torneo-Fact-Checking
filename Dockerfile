# syntax=docker/dockerfile:1

# ── Stage 1: base ─────────────────────────────────────────────────────────────
FROM python:3.11-slim AS base

# Keeps Python from generating .pyc files and enables stdout/stderr flushing
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# ── Stage 2: dependencies ─────────────────────────────────────────────────────
FROM base AS deps

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# ── Stage 3: runtime ──────────────────────────────────────────────────────────
FROM deps AS runtime

# Copy application code
COPY app.py ./

# Copy the fine-tuned model (must be present at build time or mounted as a volume)
# To build without embedding the model, use the volume approach in docker-compose.yml
COPY model/ ./model/

# Inference server port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
