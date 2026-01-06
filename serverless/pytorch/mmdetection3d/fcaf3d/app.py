#!/usr/bin/env python3
"""
FCAF3D 3D Object Detection Web Service
=====================================

This service provides REST API endpoints for FCAF3D 3D object detection
using MMDetection3D. It processes point cloud data and returns CVAT-compatible
cuboid detections.

Endpoints:
- GET /health: Health check
- POST /detect: Run 3D object detection on point cloud data

Usage:
    python app.py

Or with uvicorn:
    uvicorn app:app --host 0.0.0.0 --port 8000
"""

import os
import sys
import base64
import json
import logging
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Import the FCAF3D detector
import sys
sys.path.append('/opt/fcaf3d')
from main import FCAF3DDetector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="FCAF3D 3D Object Detection Service",
    description="REST API for FCAF3D 3D object detection using MMDetection3D",
    version="1.0.0"
)

# Global detector instance
detector = None

class DetectionRequest(BaseModel):
    """Request model for detection endpoint."""
    image: str  # Base64-encoded point cloud data
    threshold: Optional[float] = None
    frame_number: Optional[int] = 0
    job_id: Optional[str] = None
    task_id: Optional[str] = None

class DetectionResponse(BaseModel):
    """Response model for detection endpoint."""
    detections: List[Dict[str, Any]]
    frame_number: Optional[int] = None
    processing_time: Optional[float] = None

@app.on_event("startup")
async def startup_event():
    """Initialize the FCAF3D detector on startup."""
    global detector
    logger.info("🚀 Starting FCAF3D service initialization...")

    try:
        logger.info("🏗️ Creating FCAF3D detector instance...")
        detector = FCAF3DDetector(logger)
        logger.info("✅ FCAF3D detector ready for inference!")
    except Exception as e:
        logger.error(f"❌ CRITICAL: Failed to initialize FCAF3D detector: {e}")
        import traceback
        logger.error(f"Full initialization error:\n{traceback.format_exc()}")
        raise

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if detector is None or detector.model is None:
        raise HTTPException(status_code=503, detail="Service not ready - detector not initialized")

    # Try to get device info safely
    device_info = "unknown"
    try:
        # For MMDetection3D models, device might not be directly accessible
        if hasattr(detector.model, 'device'):
            device_info = str(detector.model.device)
        else:
            device_info = "model_loaded"
    except:
        device_info = "model_loaded"

    return {
        "status": "healthy",
        "service": "FCAF3D 3D Object Detection",
        "model_loaded": True,
        "device": device_info
    }

@app.post("/detect", response_model=DetectionResponse)
async def detect_objects(request: DetectionRequest):
    """
    Run 3D object detection on point cloud data.

    Args:
        request: Detection request with base64-encoded point cloud

    Returns:
        Detection results in CVAT-compatible format
    """
    import time
    start_time = time.time()

    logger.info(f"🎯 Processing detection request for frame {request.frame_number}")

    if detector is None or detector.model is None:
        logger.error("❌ Detector not initialized")
        raise HTTPException(status_code=503, detail="Service not ready - detector not initialized")

    try:
        # Decode base64 point cloud data
        logger.info("🔓 Decoding base64 point cloud data...")
        cloud_bytes = base64.b64decode(request.image)
        logger.info(f"📊 Point cloud size: {len(cloud_bytes)} bytes")

        # DEBUG: Check header
        header = cloud_bytes[:20]
        logger.info(f"🔍 DEBUG: Data header: {header!r}")

        # Run detection
        logger.info("🎯 Starting FCAF3D detection...")
        threshold = request.threshold if request.threshold is not None else detector.confidence_threshold

        detections = detector.infer(
            cloud_bytes=cloud_bytes,
            threshold=threshold,
            frame_id=request.frame_number or 0,
        )

        processing_time = time.time() - start_time

        logger.info(f"✅ Detection completed in {processing_time:.3f}s!")
        logger.info(f"🔍 Found {len(detections)} detections")

        return DetectionResponse(
            detections=detections,
            frame_number=request.frame_number,
            processing_time=round(processing_time, 3)
        )

    except Exception as e:
        logger.error(f"❌ Detection failed: {e}")
        import traceback
        logger.error(f"Full traceback:\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint with service information."""
    return {
        "service": "FCAF3D 3D Object Detection Service",
        "version": "1.0.0",
        "description": "REST API for 3D cuboid detection from point clouds",
        "endpoints": {
            "GET /health": "Health check",
            "POST /detect": "Run 3D object detection",
            "GET /": "Service information"
        }
    }

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")

    logger.info(f"🚀 Starting FCAF3D service on {host}:{port}")
    uvicorn.run(app, host=host, port=port)
