# Sign Language Translator - Dockerfile
# Multi-stage build for optimized image size

# =============================================================================
# Stage 1: Builder
# =============================================================================
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --user -r requirements.txt

# =============================================================================
# Stage 2: Runtime
# =============================================================================
FROM python:3.11-slim

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libportaudio2 \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /root/.local /root/.local

# Make sure scripts in .local are usable
ENV PATH=/root/.local/bin:$PATH

# Copy application code
COPY . .

# Set working directory to UI
WORKDIR /app/UI

# Expose port
EXPOSE 5000

# Set environment variables
ENV FLASK_APP=app.py
ENV FLASK_ENV=production
ENV PYTHONUNBUFFERED=1

# Health check (HTTP by default; override if using HTTPS)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -fk http://localhost:${PORT:-5000}/health || curl -fk https://localhost:${PORT:-5000}/health || exit 1

# Run via start.sh so HOST/PORT/TRAFFIC/SSL_* from .env are respected
RUN chmod +x /app/UI/start.sh
CMD ["./start.sh"]
