# Production Dockerfile for GlucoTwin Backend Service
FROM python:3.12-slim AS base

# Prevent Python from writing .pyc and buffer stdout/stderr
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    FORCE_CPU=1 \
    KMP_DUPLICATE_LIB_OK=TRUE \
    PORT=8000

# Install curl for healthcheck and system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies in a single cached layer
# Pre-install CPU-only PyTorch to minimize image footprint and download time
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
    torch==2.2.2+cpu

# Copy requirements and install remaining python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code, configurations, model checkpoint bundle, and database
COPY pyproject.toml .
COPY gluco_twin ./gluco_twin
COPY models/current ./models/current
COPY data/glucotwin.db ./data/glucotwin.db

# Install package in editable/develop mode
RUN pip install --no-cache-dir -e .

# Create non-root user for security compliance
RUN groupadd -r appuser && useradd -r -g appuser appuser \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

# Container healthcheck
HEALTHCHECK --interval=20s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Launch production server with Uvicorn
CMD ["uvicorn", "gluco_twin.service.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
