#!/usr/bin/env python3
"""
MediaPipe Pose Detection Service for CVAT
==========================================

This service provides pose detection capabilities using MediaPipe,
optimized for egocentric vision and hand-focused pose estimation.

Features:
- 33-point pose estimation
- Hand-focused filtering
- RESTful API compatible with CVAT
- Optimized for CPU inference
- Configurable confidence thresholds

Usage:
    python app.py

API Endpoints:
    POST /detect - Detect poses in image
    GET /health - Health check
"""

import base64
import io
import json
import logging
import os
from typing import Dict, List, Optional, Any

import cv2
import mediapipe as mp
import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from PIL import Image
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize MediaPipe Pose and Hands
mp_pose = mp.tasks.vision.PoseLandmarker
mp_hands = mp.tasks.vision.HandLandmarker
mp_vision = mp.tasks.vision

# Global detectors (initialized on startup)
pose_detector = None
hands_detector = None

def init_detectors():
    """Initialize MediaPipe pose and hands detectors with optimized settings."""
    global pose_detector, hands_detector

    # Initialize Pose detector
    if pose_detector is None:
        logger.info("Initializing MediaPipe Pose detector...")

        # Download pose landmarker model if not present
        pose_model_path = "/tmp/pose_landmarker_lite.task"
        if not os.path.exists(pose_model_path):
            logger.info("Downloading pose landmarker model...")
            import urllib.request
            pose_url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
            urllib.request.urlretrieve(pose_url, pose_model_path)
            logger.info("Pose model downloaded successfully")

        # Create pose landmarker options
        pose_options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=pose_model_path
            ),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_segmentation_masks=False
        )

        # Create the pose landmarker
        pose_detector = mp_pose.create_from_options(pose_options)
        logger.info("MediaPipe Pose detector initialized successfully")

    # Initialize Hands detector
    if hands_detector is None:
        logger.info("Initializing MediaPipe Hands detector...")

        # Download hand landmarker model if not present
        hands_model_path = "/tmp/hand_landmarker.task"
        if not os.path.exists(hands_model_path):
            logger.info("Downloading hand landmarker model...")
            import urllib.request
            hands_url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
            urllib.request.urlretrieve(hands_url, hands_model_path)
            logger.info("Hand model downloaded successfully")

        # Create hand landmarker options
        hands_options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=hands_model_path
            ),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_hands=2,  # Detect up to 2 hands
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Create the hand landmarker
        hands_detector = mp_hands.create_from_options(hands_options)
        logger.info("MediaPipe Hands detector initialized successfully")

    return pose_detector, hands_detector

def image_to_cv2(image_data: bytes) -> np.ndarray:
    """Convert image bytes to OpenCV format."""
    try:
        # Try to decode as base64 first (CVAT format)
        try:
            image_bytes = base64.b64decode(image_data)
        except:
            # If not base64, treat as raw image bytes
            image_bytes = image_data

        # Convert to PIL Image first, then to OpenCV
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        cv2_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        return cv2_image
    except Exception as e:
        logger.error(f"Failed to decode image: {e}")
        raise ValueError(f"Invalid image format: {e}")

def process_combined_results(pose_results, hands_results, image_height: int, image_width: int, threshold: float = 0.5) -> List[Dict]:
    """
    Process combined MediaPipe Pose and Hands results into CVAT-compatible format.

    MediaPipe Pose keypoints (33 points):
    0: nose, 1: left_eye_inner, 2: left_eye, 3: left_eye_outer, 4: right_eye_inner,
    5: right_eye, 6: right_eye_outer, 7: left_ear, 8: right_ear, 9: mouth_left,
    10: mouth_right, 11: left_shoulder, 12: right_shoulder, 13: left_elbow,
    14: right_elbow, 15: left_wrist, 16: right_wrist,

    Hand keypoints (finger bases from Pose):
    17: left_pinky_base, 18: right_pinky_base,
    19: left_index_base, 20: right_index_base,
    21: left_thumb_base, 22: right_thumb_base,

    Body keypoints:
    23: left_hip, 24: right_hip, 25: left_knee, 26: right_knee,
    27: left_ankle, 28: right_ankle, 29: left_heel, 30: right_heel,
    31: left_foot_index, 32: right_foot_index

    MediaPipe Hands keypoints (21 per hand):
    0: wrist, 1: thumb_cmc, 2: thumb_mcp, 3: thumb_ip, 4: thumb_tip,
    5: index_mcp, 6: index_pip, 7: index_dip, 8: index_tip,
    9: middle_mcp, 10: middle_pip, 11: middle_dip, 12: middle_tip,
    13: ring_mcp, 14: ring_pip, 15: ring_dip, 16: ring_tip,
    17: pinky_mcp, 18: pinky_pip, 19: pinky_dip, 20: pinky_tip
    """
    skeleton = {
        "label": "person",
        "type": "skeleton",
        "elements": []
    }

    # Process Pose results (body + basic hand keypoints)
    if pose_results and pose_results.pose_landmarks:
        pose_landmarks = pose_results.pose_landmarks[0]

        # CVAT skeleton format mapping for pose keypoints
        pose_keypoints = {
            # Face
            0: "nose", 1: "left_eye", 2: "right_eye", 3: "left_ear", 4: "right_ear",
            # Upper body
            5: "left_shoulder", 6: "right_shoulder", 7: "left_elbow", 8: "right_elbow",
            9: "left_wrist", 10: "right_wrist",
            # Lower body
            11: "left_hip", 12: "right_hip", 13: "left_knee", 14: "right_knee",
            15: "left_ankle", 16: "right_ankle"
        }

        # Add pose keypoints (always include all keypoints, even low confidence ones)
        for mp_idx, keypoint_name in pose_keypoints.items():
            landmark = pose_landmarks[mp_idx]
            confidence = getattr(landmark, 'visibility', None) or 1.0

            element = {
                "label": keypoint_name,
                "type": "points",
                "outside": confidence <= threshold,
                "points": [
                    landmark.x * image_width,
                    landmark.y * image_height
                ],
                "attributes": [
                    {"name": "confidence", "value": str(confidence)}
                ]
            }
            skeleton["elements"].append(element)

    # Process Hands results (detailed finger keypoints)
    if hands_results and hands_results.hand_landmarks:
        # Hand landmark names
        hand_keypoints = [
            "wrist", "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
            "index_mcp", "index_pip", "index_dip", "index_tip",
            "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
            "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
            "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip"
        ]

        # Process each detected hand
        for hand_idx, hand_landmarks in enumerate(hands_results.hand_landmarks):
            # Use handedness classification if available, otherwise assume left/right based on index
            try:
                if hasattr(hands_results, 'handedness') and hands_results.handedness and hand_idx < len(hands_results.handedness):
                    handedness = hands_results.handedness[hand_idx][0].category_name.lower()
                elif hasattr(hands_results, 'multi_handedness') and hands_results.multi_handedness and hand_idx < len(hands_results.multi_handedness):
                    handedness = hands_results.multi_handedness[hand_idx].classification[0].label.lower()
                else:
                    handedness = "left" if hand_idx == 0 else "right"
            except (AttributeError, IndexError, KeyError):
                handedness = "left" if hand_idx == 0 else "right"

            # Add hand keypoints with handedness prefix (always include all keypoints)
            for kp_idx, landmark in enumerate(hand_landmarks):
                confidence = getattr(landmark, 'visibility', None) or 1.0

                keypoint_name = f"{handedness}_{hand_keypoints[kp_idx]}"
                element = {
                    "label": keypoint_name,
                    "type": "points",
                    "outside": confidence <= threshold,
                    "points": [
                        landmark.x * image_width,
                        landmark.y * image_height
                    ],
                    "attributes": [
                        {"name": "confidence", "value": str(confidence)}
                    ]
                }
                skeleton["elements"].append(element)

    # For egocentric videos, we want to include all keypoints but mark low-confidence ones as outside
    # Count visible keypoints (those with confidence above threshold)
    visible_keypoints = [elem for elem in skeleton["elements"] if not elem["outside"]]

    # Count pose vs hand keypoints for logging
    hand_labels = {f"{side}_{finger}_{joint}" for side in ["left", "right"]
                   for finger in ["wrist", "thumb", "index", "middle", "ring", "pinky"]
                   for joint in ["cmc", "mcp", "pip", "dip", "tip"] if joint != "cmc" or finger == "thumb"}
    hand_labels.update([f"{side}_wrist" for side in ["left", "right"]])

    pose_keypoints = [elem for elem in skeleton["elements"] if elem['label'] not in hand_labels]
    hand_keypoints = [elem for elem in skeleton["elements"] if elem['label'] in hand_labels]

    # Require at least 2 visible keypoints for a valid skeleton
    if len(visible_keypoints) < 2:
        logger.info(f"Insufficient visible keypoints detected ({len(visible_keypoints)}), skipping")
        return []

    logger.info(f"Detected skeleton with {len([kp for kp in pose_keypoints if not kp['outside']])} visible pose keypoints and {len([kp for kp in hand_keypoints if not kp['outside']])} visible hand keypoints")

    logger.info(f"Processed combined pose+hands with {len(skeleton['elements'])} keypoints")
    return [skeleton]

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application startup and shutdown."""
    # Startup
    init_detectors()
    yield
    # Shutdown (if needed)
    pass

app = FastAPI(
    title="MediaPipe Pose + Hands Service",
    description="Real-time pose estimation with hand tracking for egocentric vision",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "mediapipe-pose"}

@app.post("/detect")
async def detect_pose(
    data: Dict[str, Any]
):
    """
    Detect poses in image.

    Args:
        data: JSON object with image and parameters
            - image: Base64 encoded image (required)
            - threshold: Confidence threshold for keypoints (optional, default: 0.3)

    Returns:
        CVAT-compatible skeleton annotations
    """
    try:
        # Debug logging
        logger.info(f"Received request data keys: {list(data.keys()) if isinstance(data, dict) else type(data)}")

        # Extract parameters from JSON
        image_b64 = data.get('image')
        threshold = data.get('threshold', 0.05)  # Very low threshold for egocentric videos

        logger.info(f"Image provided: {image_b64 is not None}, threshold: {threshold}")

        if not image_b64:
            logger.error("No image provided in request")
            raise HTTPException(status_code=400, detail="No image provided")

        image_data = image_b64.encode('utf-8')
        logger.info(f"Image data length: {len(image_data)}")

        # Convert to OpenCV format
        cv2_image = image_to_cv2(image_data)
        image_height, image_width = cv2_image.shape[:2]

        # Get detectors
        pose_detector, hands_detector = init_detectors()

        # Create MediaPipe Image object
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2_image)

        # Process image with both detectors
        logger.info(f"Processing image: {image_width}x{image_height}")

        # Run pose detection
        pose_results = pose_detector.detect(mp_image)

        # Run hands detection
        hands_results = hands_detector.detect(mp_image)

        # Debug: Log detection results
        pose_detected = pose_results and pose_results.pose_landmarks
        hands_detected = hands_results and hands_results.hand_landmarks

        logger.info(f"Pose detected: {pose_detected}, Hands detected: {hands_detected}")
        if pose_detected:
            logger.info(f"Pose landmarks count: {len(pose_results.pose_landmarks)}")
        if hands_detected:
            logger.info(f"Hand landmarks count: {len(hands_results.hand_landmarks)}")

        # Combine results
        skeletons = process_combined_results(pose_results, hands_results, image_height, image_width, threshold)

        logger.info(f"Detected {len(skeletons)} poses")
        return JSONResponse(content=skeletons)

    except Exception as e:
        logger.error(f"Error processing pose detection: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    """Service information."""
    return {
        "service": "MediaPipe Pose Detection",
        "version": "1.0.0",
        "description": "Pose estimation service optimized for egocentric vision",
        "endpoints": {
            "GET /health": "Health check",
            "POST /detect": "Detect poses in image",
            "GET /": "Service information"
        },
        "usage": {
            "threshold": "Confidence threshold (0.0-1.0, default: 0.3)",
            "image": "Base64 encoded image string",
            "image_file": "Direct image file upload (alternative)"
        }
    }

def check_port_available(host: str, port: int) -> bool:
    """Check if a port is available."""
    import socket
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex((host, port))
        sock.close()
        return result != 0  # 0 means connection successful (port in use)
    except:
        return True

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")

    # Check if port is available, if not try next available port
    if not check_port_available(host, port):
        logger.warning(f"Port {port} is already in use, trying {port + 1}")
        port += 1
        if not check_port_available(host, port):
            logger.error(f"Ports {port - 1} and {port} are both in use. Please specify a different port with PORT environment variable.")
            exit(1)

    logger.info(f"Starting MediaPipe Pose Service on {host}:{port}")
    uvicorn.run(app, host=host, port=port)
