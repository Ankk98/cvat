# MediaPipe Pose Detection Service - Docker Image
# ================================================
#
# This Dockerfile creates a containerized version of the MediaPipe pose service.
#
# Build:
#   docker build -t mediapipe-pose-service .
#
# Run:
#   docker run -p 8000:8000 mediapipe-pose-service

FROM python:3.10-slim-bookworm

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies (minimal set for MediaPipe CPU)
RUN apt-get update && apt-get install -y \
    curl \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download MediaPipe models during build (avoids runtime DNS issues)
RUN mkdir -p /tmp && \
    curl -L -o /tmp/pose_landmarker_lite.task \
    https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task && \
    curl -L -o /tmp/hand_landmarker.task \
    https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task && \
    chmod 644 /tmp/*.task

# Copy application code
COPY app.py .

# Create non-root user
RUN useradd --create-home --shell /bin/bash app && \
    chown -R app:app /app && \
    chown -R app:app /tmp/*.task
USER app

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["python", "app.py"]
