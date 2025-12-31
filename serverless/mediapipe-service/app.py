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
import time
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

# Global stateless detectors for IMAGE mode (interactive/random access)
image_pose_detector = None
image_hands_detector = None

def get_image_mode_detectors():
    """Get global stateless detectors for IMAGE mode."""
    global image_pose_detector, image_hands_detector

    if image_pose_detector is None:
        logger.info("Initializing MediaPipe IMAGE mode detectors...")

        # Use pre-downloaded pose landmarker model
        pose_model_path = "/tmp/pose_landmarker_lite.task"
        if not os.path.exists(pose_model_path):
            import urllib.request
            pose_url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
            urllib.request.urlretrieve(pose_url, pose_model_path)

        # IMAGE mode options
        pose_options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=pose_model_path),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.3,
            min_pose_presence_confidence=0.3,
            min_tracking_confidence=0.3,
            output_segmentation_masks=False
        )
        image_pose_detector = mp_pose.create_from_options(pose_options)

        # Use pre-downloaded hand landmarker model
        hands_model_path = "/tmp/hand_landmarker.task"
        if not os.path.exists(hands_model_path):
            import urllib.request
            hands_url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
            urllib.request.urlretrieve(hands_url, hands_model_path)

        # IMAGE mode options
        hands_options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=hands_model_path),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=0.3,
            min_hand_presence_confidence=0.3,
            min_tracking_confidence=0.3
        )
        image_hands_detector = mp_hands.create_from_options(hands_options)

    return image_pose_detector, image_hands_detector

# Session management
class DetectorSession:
    def __init__(self):
        self.pose_detector = None
        self.hands_detector = None
        self.last_access = time.time()
        self.last_timestamp = -1

    def close(self):
        if self.pose_detector:
            self.pose_detector.close()
            self.pose_detector = None
        if self.hands_detector:
            self.hands_detector.close()
            self.hands_detector = None

sessions: Dict[str, DetectorSession] = {}

def get_session_detectors(session_id: str):
    """Get or create detectors for a specific session."""
    global sessions

    # Cleanup expired sessions (older than 10 minutes)
    now = time.time()
    expired = [k for k, v in sessions.items() if now - v.last_access > 600]
    for k in expired:
        logger.info(f"Closing expired session {k}")
        sessions[k].close()
        del sessions[k]

    # Create new session if needed
    if session_id not in sessions:
        logger.info(f"Creating new session {session_id}")
        sessions[session_id] = DetectorSession()

        # Initialize detectors for this session
        try:
            # Use pre-downloaded pose landmarker model (downloaded during Docker build)
            pose_model_path = "/tmp/pose_landmarker_lite.task"
            if not os.path.exists(pose_model_path):
                logger.warning("Pose model not found at /tmp/pose_landmarker_lite.task, attempting download...")
                import urllib.request
                pose_url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
                urllib.request.urlretrieve(pose_url, pose_model_path)

            # Create pose landmarker options
            pose_options = mp.tasks.vision.PoseLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(
                    model_asset_path=pose_model_path
                ),
                running_mode=mp.tasks.vision.RunningMode.VIDEO,
                num_poses=1,
                min_pose_detection_confidence=0.3,
                min_pose_presence_confidence=0.3,
                min_tracking_confidence=0.3,
                output_segmentation_masks=False
            )
            sessions[session_id].pose_detector = mp_pose.create_from_options(pose_options)

            # Use pre-downloaded hand landmarker model
            hands_model_path = "/tmp/hand_landmarker.task"
            if not os.path.exists(hands_model_path):
                logger.warning("Hand model not found at /tmp/hand_landmarker.task, attempting download...")
                import urllib.request
                hands_url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
                urllib.request.urlretrieve(hands_url, hands_model_path)

            # Create hand landmarker options
            hands_options = mp.tasks.vision.HandLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(
                    model_asset_path=hands_model_path
                ),
                running_mode=mp.tasks.vision.RunningMode.VIDEO,
                num_hands=2,
                min_hand_detection_confidence=0.3,
                min_hand_presence_confidence=0.3,
                min_tracking_confidence=0.3
            )
            sessions[session_id].hands_detector = mp_hands.create_from_options(hands_options)

            logger.info(f"Initialized detectors for session {session_id}")

        except Exception as e:
            logger.error(f"Failed to initialize detectors for session {session_id}: {e}")
            if session_id in sessions:
                sessions[session_id].close()
                del sessions[session_id]
            raise

    # Update access time
    sessions[session_id].last_access = now

    return sessions[session_id].pose_detector, sessions[session_id].hands_detector

def prioritize_hands(hands_results, image_width: int, image_height: int, center_weight: float = 0.5) -> List:
    """
    Prioritize hands based on:
    - Distance from center of frame (egocentric videos focus on center)
    - Number of visible keypoints
    - Average confidence

    Returns: List of prioritized hand landmarks (up to 2)
    """
    if not hands_results or not hands_results.hand_landmarks:
        return []

    # Calculate center of frame
    center_x, center_y = image_width / 2, image_height / 2

    # Score each hand
    scored_hands = []
    for hand_idx, hand_landmarks in enumerate(hands_results.hand_landmarks):
        # Calculate distance from center using wrist (first keypoint)
        wrist = hand_landmarks[0]
        wrist_x = wrist.x * image_width
        wrist_y = wrist.y * image_height

        dist_from_center = np.sqrt(
            (wrist_x - center_x)**2 +
            (wrist_y - center_y)**2
        )
        max_dim = max(image_width, image_height)
        normalized_dist = dist_from_center / max_dim if max_dim > 0 else 1.0

        # Count visible keypoints (hand landmarks don't have visibility, so count all)
        # For hand landmarks, visibility/presence may be None, so we assume all are visible
        def get_hand_confidence(lm):
            vis = getattr(lm, 'visibility', None)
            pres = getattr(lm, 'presence', None)
            # If visibility/presence exist, use them; otherwise assume 1.0 (visible)
            if vis is not None:
                return vis
            elif pres is not None:
                return pres
            else:
                return 1.0

        visible_count = sum(1 for lm in hand_landmarks
                          if get_hand_confidence(lm) > 0.3)

        # Average confidence
        avg_confidence = np.mean([get_hand_confidence(lm)
                                 for lm in hand_landmarks])

        # Combined score (lower distance = higher score)
        # Center proximity: 50%, visible keypoints: 30%, confidence: 20%
        score = (1 - normalized_dist) * center_weight + \
                (visible_count / 21) * 0.3 + \
                avg_confidence * 0.2

        scored_hands.append((score, hand_idx, hand_landmarks))

    # Sort by score (highest first)
    scored_hands.sort(reverse=True, key=lambda x: x[0])

    # Return top 2 hands (or all if less than 2)
    return [hand for _, _, hand in scored_hands[:2]]

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
        # Convert PIL RGB to numpy array, then to OpenCV BGR
        # MediaPipe expects RGB format, but we convert to BGR for OpenCV compatibility
        # Then MediaPipe Image will convert back to RGB internally
        cv2_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        return cv2_image
    except Exception as e:
        logger.error(f"Failed to decode image: {e}")
        raise ValueError(f"Invalid image format: {e}")

def process_combined_results(pose_results, hands_results, image_height: int, image_width: int, threshold: float = 0.5, prioritized_hands: Optional[List] = None) -> List[Dict]:
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
    # Determine which label to use based on what's detected
    # We'll create skeletons for both labels if applicable

    # Always create person-skeleton if pose is detected (body + hands)
    person_skeleton = {
        "label": "person-skeleton",
        "type": "skeleton",
        "elements": [],
        "points": []
    }

    # Create hands-skeleton separately (hands only, no body)
    hands_skeleton = {
        "label": "hands-skeleton",
        "type": "skeleton",
        "elements": [],
        "points": []
    }

    # Create hands-shoulders-skeleton (shoulders, elbows, wrists + hands)
    hands_shoulders_skeleton = {
        "label": "hands-shoulders-skeleton",
        "type": "skeleton",
        "elements": [],
        "points": []
    }

    # Process Pose results (body + basic hand keypoints)
    if pose_results and pose_results.pose_landmarks:
        pose_landmarks = pose_results.pose_landmarks[0]

        # CVAT skeleton format mapping for pose keypoints (corrected MediaPipe indices)
        pose_keypoints = {
            # Face
            0: "nose", 2: "left_eye", 5: "right_eye", 7: "left_ear", 8: "right_ear",
            # Upper body
            11: "left_shoulder", 12: "right_shoulder", 13: "left_elbow", 14: "right_elbow",
            15: "left_wrist", 16: "right_wrist",
            # Lower body
            23: "left_hip", 24: "right_hip", 25: "left_knee", 26: "right_knee",
            27: "left_ankle", 28: "right_ankle"
        }

        # Upper body keypoints for hands-shoulders-skeleton (shoulders, elbows, wrists only - corrected indices)
        upper_body_keypoints = {
            11: "left_shoulder", 12: "right_shoulder", 13: "left_elbow", 14: "right_elbow",
            15: "left_wrist", 16: "right_wrist"
        }

        # Add pose keypoints (always include all keypoints, even low confidence ones)
        for mp_idx, keypoint_name in pose_keypoints.items():
            landmark = pose_landmarks[mp_idx]
            confidence = getattr(landmark, 'visibility', None) or 1.0

            # Convert normalized coordinates to pixel coordinates
            # MediaPipe: (0,0) = top-left, (1,1) = bottom-right
            # CVAT expects: [x, y] in pixel coordinates
            # IMPORTANT: Clamp coordinates to valid range [0, 1] as MediaPipe can return values > 1.0
            x_norm = max(0.0, min(1.0, landmark.x))
            y_norm = max(0.0, min(1.0, landmark.y))

            x_pixel = x_norm * image_width
            y_pixel = y_norm * image_height

            # Log first few keypoints for debugging
            if len(person_skeleton["elements"]) < 3:
                if landmark.x != x_norm or landmark.y != y_norm:
                    logger.info(f"Pose keypoint {keypoint_name}: normalized=({landmark.x:.3f}, {landmark.y:.3f}) -> clamped=({x_norm:.3f}, {y_norm:.3f}), pixel=({x_pixel:.1f}, {y_pixel:.1f})")
                else:
                    logger.info(f"Pose keypoint {keypoint_name}: normalized=({landmark.x:.3f}, {landmark.y:.3f}), pixel=({x_pixel:.1f}, {y_pixel:.1f})")

            # Warn if coordinates are invalid or in bottom-right corner
            if landmark.x > 1.0 or landmark.y > 1.0:
                logger.warning(f"⚠️  Keypoint {keypoint_name} has invalid normalized coordinates (>1.0): ({landmark.x:.3f}, {landmark.y:.3f}) - clamping to valid range")
            elif landmark.x > 0.9 and landmark.y > 0.8:
                logger.warning(f"⚠️  Keypoint {keypoint_name} detected at bottom-right corner: normalized=({landmark.x:.3f}, {landmark.y:.3f}), pixel=({x_pixel:.1f}, {y_pixel:.1f})")

            element = {
                "label": keypoint_name,
                "type": "points",
                "outside": confidence <= threshold,
                "points": [
                    x_pixel,
                    y_pixel
                ],
                "attributes": [
                    {"name": "confidence", "value": str(confidence)}
                ]
            }
            person_skeleton["elements"].append(element)

            # Also add upper body keypoints (shoulders, elbows, wrists) to hands-shoulders-skeleton
            if keypoint_name in upper_body_keypoints.values():
                hands_shoulders_skeleton["elements"].append(element)

    # Process Hands results (detailed finger keypoints)
    # Use prioritized hands if provided, otherwise use all detected hands
    hands_to_process = prioritized_hands if prioritized_hands else (hands_results.hand_landmarks if (hands_results and hands_results.hand_landmarks) else [])

    if hands_to_process:
        # Hand landmark names
        hand_keypoints = [
            "wrist", "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
            "index_mcp", "index_pip", "index_dip", "index_tip",
            "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
            "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
            "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip"
        ]

        # Process prioritized hands (up to 2, closest to center)
        for hand_idx, hand_landmarks in enumerate(hands_to_process):
            # Use handedness classification if available
            # Note: Since we're using prioritized hands, we need to map back to original indices
            # For simplicity, use position-based handedness (first = left, second = right)
            # This is acceptable for egocentric videos where handedness may be ambiguous
            try:
                # Try to get handedness from original results if available
                original_idx = None
                if hasattr(hands_results, 'handedness') and hands_results.handedness:
                    # Map prioritized hand back to original (simplified: use index)
                    if hand_idx < len(hands_results.handedness):
                        handedness = hands_results.handedness[hand_idx][0].category_name.lower()
                    else:
                        handedness = "left" if hand_idx == 0 else "right"
                elif hasattr(hands_results, 'multi_handedness') and hands_results.multi_handedness:
                    if hand_idx < len(hands_results.multi_handedness):
                        handedness = hands_results.multi_handedness[hand_idx].classification[0].label.lower()
                    else:
                        handedness = "left" if hand_idx == 0 else "right"
                else:
                    handedness = "left" if hand_idx == 0 else "right"
            except (AttributeError, IndexError, KeyError):
                handedness = "left" if hand_idx == 0 else "right"

            # Add hand keypoints with handedness prefix (always include all keypoints)
            for kp_idx, landmark in enumerate(hand_landmarks):
                # Hand landmarks may not have visibility/presence attributes
                vis = getattr(landmark, 'visibility', None)
                pres = getattr(landmark, 'presence', None)
                if vis is not None:
                    confidence = vis
                elif pres is not None:
                    confidence = pres
                else:
                    confidence = 1.0  # Assume visible if no confidence metric available

                keypoint_name = f"{handedness}_{hand_keypoints[kp_idx]}"

                # Convert normalized coordinates to pixel coordinates
                # MediaPipe: (0,0) = top-left, (1,1) = bottom-right
                # CVAT expects: [x, y] in pixel coordinates
                # IMPORTANT: Clamp coordinates to valid range [0, 1] as MediaPipe can return values > 1.0
                x_norm = max(0.0, min(1.0, landmark.x))
                y_norm = max(0.0, min(1.0, landmark.y))

                x_pixel = x_norm * image_width
                y_pixel = y_norm * image_height

                # Log first few hand keypoints for debugging
                if len([e for e in hands_skeleton["elements"] if 'hand' in e.get('label', '').lower() or 'wrist' in e.get('label', '').lower()]) < 3:
                    if landmark.x != x_norm or landmark.y != y_norm:
                        logger.info(f"Hand keypoint {keypoint_name}: normalized=({landmark.x:.3f}, {landmark.y:.3f}) -> clamped=({x_norm:.3f}, {y_norm:.3f}), pixel=({x_pixel:.1f}, {y_pixel:.1f})")
                    else:
                        logger.info(f"Hand keypoint {keypoint_name}: normalized=({landmark.x:.3f}, {landmark.y:.3f}), pixel=({x_pixel:.1f}, {y_pixel:.1f})")

                element = {
                    "label": keypoint_name,
                    "type": "points",
                    "outside": confidence <= threshold,
                    "points": [
                        x_pixel,
                        y_pixel
                    ],
                    "attributes": [
                        {"name": "confidence", "value": str(confidence)}
                    ]
                }
                # Add to person-skeleton (body + hands) and hands-skeleton (hands only)
                # For person-skeleton, avoid duplicate wrists (use the ones from pose)
                if keypoint_name not in ["left_wrist", "right_wrist"]:
                    person_skeleton["elements"].append(element)
                hands_skeleton["elements"].append(element)

                # Add to hands-shoulders-skeleton (shoulders + hands), but skip wrist since it's already in upper body
                # In hands-shoulders-skeleton, wrist comes from pose (upper body), not from hands
                if keypoint_name not in ["left_wrist", "right_wrist"]:
                    hands_shoulders_skeleton["elements"].append(element)

    # For egocentric videos, we want to include all keypoints but mark low-confidence ones as outside
    # Count visible keypoints (those with confidence above threshold)
    visible_keypoints = [elem for elem in person_skeleton["elements"] if not elem["outside"]]

    # Count pose vs hand keypoints for logging
    hand_labels = {f"{side}_{finger}_{joint}" for side in ["left", "right"]
                   for finger in ["wrist", "thumb", "index", "middle", "ring", "pinky"]
                   for joint in ["cmc", "mcp", "pip", "dip", "tip"] if joint != "cmc" or finger == "thumb"}
    hand_labels.update([f"{side}_wrist" for side in ["left", "right"]])

    pose_keypoints = [elem for elem in person_skeleton["elements"] if elem['label'] not in hand_labels]
    hand_keypoints = [elem for elem in person_skeleton["elements"] if elem['label'] in hand_labels]

    # Relaxed filtering logic for egocentric videos:
    # - If hands detected: require at least 3 hand keypoints
    # - If only pose detected: require at least 5 pose keypoints
    # - Otherwise: require at least 2 visible keypoints
    visible_hand_kps = [kp for kp in hand_keypoints if not kp['outside']]
    visible_pose_kps = [kp for kp in pose_keypoints if not kp['outside']]

    if len(hands_to_process) > 0:
        # Hands detected: require at least 3 hand keypoints
        if len(visible_hand_kps) < 3:
            logger.info(f"Insufficient visible hand keypoints detected ({len(visible_hand_kps)}), need at least 3, skipping")
            return []
    elif len(pose_keypoints) > 0:
        # Only pose detected: require at least 5 pose keypoints
        if len(visible_pose_kps) < 5:
            logger.info(f"Insufficient visible pose keypoints detected ({len(visible_pose_kps)}), need at least 5, skipping")
            return []
    else:
        # Fallback: require at least 2 visible keypoints
        if len(visible_keypoints) < 2:
            logger.info(f"Insufficient visible keypoints detected ({len(visible_keypoints)}), skipping")
            return []

    logger.info(f"Detected skeleton with {len([kp for kp in pose_keypoints if not kp['outside']])} visible pose keypoints and {len([kp for kp in hand_keypoints if not kp['outside']])} visible hand keypoints")

    # Finalize skeletons by setting points to empty list
    # CVAT backend expects empty points for skeleton type in some versions.
    for skeleton in [person_skeleton, hands_shoulders_skeleton, hands_skeleton]:
        skeleton["points"] = []

    result_skeletons = []

    # Always return person-skeleton if we have body keypoints (with or without hands)
    if len(visible_pose_kps) > 0 or len(visible_hand_kps) > 0:
        logger.info(f"Adding person-skeleton with {len(person_skeleton['elements'])} keypoints")
        result_skeletons.append(person_skeleton)

    # Return hands-shoulders-skeleton
    if len(hands_shoulders_skeleton["elements"]) > 0:
        logger.info(f"Adding hands-shoulders-skeleton with {len(hands_shoulders_skeleton['elements'])} keypoints")
        result_skeletons.append(hands_shoulders_skeleton)

    # Also return hands-skeleton
    if len(visible_hand_kps) > 0 and hands_skeleton["elements"] and hands_skeleton["points"] != [0.0, 0.0, 0.0, 0.0]:
        logger.info(f"Adding hands-skeleton with {len(hands_skeleton['elements'])} keypoints")
        result_skeletons.append(hands_skeleton)

    return result_skeletons

def calculate_skeleton_bbox(elements):
    """Calculate [xtl, ytl, xbr, ybr] bounding box from skeleton elements."""
    xs = []
    ys = []
    for el in elements:
        # Only use visible elements if possible
        if not el.get("outside", False) and "points" in el:
            xs.append(el["points"][0])
            ys.append(el["points"][1])

    # If no visible elements, use all elements
    if not xs:
        for el in elements:
            if "points" in el:
                xs.append(el["points"][0])
                ys.append(el["points"][1])

    if not xs:
        return [0.0, 0.0, 0.0, 0.0]

    return [float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))]

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application startup and shutdown."""
    # Startup
    yield
    # Shutdown
    logger.info("Shutting down, closing all sessions...")
    for session in sessions.values():
        session.close()
    sessions.clear()

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

def get_shape_center(shape, width, height):
    """Calculate centroid of a shape (skeleton or rectangle)."""
    if not shape:
        return None

    # Handle list input (e.g. bounding box [xmin, ymin, xmax, ymax])
    if isinstance(shape, list):
        if len(shape) == 4:
            # Assume [xmin, ymin, xmax, ymax]
            return ((shape[0] + shape[2]) / 2, (shape[1] + shape[3]) / 2)
        else:
            logger.warning(f"get_shape_center received list with unexpected length {len(shape)}: {shape}")
            return None

    if not isinstance(shape, dict):
        logger.error(f"get_shape_center expected dict or list, got {type(shape)}: {shape}")
        return None

    points = shape.get("points", [])
    if not points and "elements" in shape:
        # Skeleton: aggregate points from elements
        elements = shape["elements"]
        if isinstance(elements, list):
            for elem in elements:
                if isinstance(elem, dict):
                    if not elem.get("outside", False):
                        points.extend(elem.get("points", []))
                else:
                    logger.warning(f"Skipping non-dict element in shape: {type(elem)}")

    if not points:
        return None

    # Points are usually [x, y, x, y...]
    xs = points[0::2]
    ys = points[1::2]

    if not xs or not ys:
        return None

    return (sum(xs) / len(xs), sum(ys) / len(ys))

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
            - frame_number: Frame number for timestamp calculation (optional, default: 0)
            - tracking_mode: 'image' or 'video' (optional, default: 'image')

    Returns:
        CVAT-compatible skeleton annotations with optional tracking info
    """
    try:
        # Debug logging
        logger.info(f"Received request with {len(data) if isinstance(data, dict) else 0} keys: {list(data.keys()) if isinstance(data, dict) else type(data)}")

        # Extract parameters from JSON
        image_b64 = data.get('image')
        threshold = data.get('threshold', 0.05)  # Very low threshold for egocentric videos
        frame_number = data.get('frame_number', 0)
        tracking_mode = data.get('tracking_mode', 'image')  # 'image' or 'video'

        logger.info(f"Image provided: {image_b64 is not None}, threshold: {threshold}, frame_number: {frame_number}, tracking_mode: {tracking_mode}")

        if not image_b64:
            logger.error("No image provided in request")
            raise HTTPException(status_code=400, detail="No image provided")

        image_data = image_b64.encode('utf-8')
        logger.info(f"Image data length: {len(image_data)}")

        # Convert to OpenCV format
        cv2_image = image_to_cv2(image_data)
        image_height, image_width = cv2_image.shape[:2]

        pose_detector = None
        hands_detector = None

        # Choose detectors based on mode
        session_id = None
        if tracking_mode == 'video':
            # Use session-based stateful detectors
            session_key = data.get('job_id') or data.get('task_id')
            if not session_key:
                session_key = "default_video_session"
                logger.warning(f"No job_id or task_id provided for video mode, using: {session_key}")

            session_id = str(session_key)
            pose_detector, hands_detector = get_session_detectors(session_id)
        else:
            # Use global stateless detectors
            pose_detector, hands_detector = get_image_mode_detectors()

        # Create MediaPipe Image object
        # MediaPipe expects RGB format, but cv2_image is BGR
        # Convert BGR to RGB for MediaPipe
        cv2_rgb = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2_rgb)

        # Process image with both detectors
        logger.info(f"Processing image: {image_width}x{image_height} (width x height) in {tracking_mode.upper()} mode")

        hands_results = None
        pose_results = None

        if tracking_mode == 'video':
            # Use detect_for_video for VIDEO mode with timestamp
            frame_number = data.get('frame_number', 0)
            timestamp_ms = int(frame_number * 1000 / 30)  # Assuming 30 FPS

            session = sessions[session_id]
            if timestamp_ms <= session.last_timestamp:
                logger.warning(f"Backward jump in timestamp (last: {session.last_timestamp}, current: {timestamp_ms}). Re-initializing detectors.")
                sessions[session_id].close()
                del sessions[session_id]
                pose_detector, hands_detector = get_session_detectors(session_id)
                session = sessions[session_id]
                logger.info(f"Re-initialized session {session_id}, pose_detector: {pose_detector is not None}, hands_detector: {hands_detector is not None}")

            session.last_timestamp = timestamp_ms

            hands_results = hands_detector.detect_for_video(mp_image, timestamp_ms)
            pose_results = pose_detector.detect_for_video(mp_image, timestamp_ms)
        else:
            # Use detect for IMAGE mode (no timestamp)
            hands_results = hands_detector.detect(mp_image)
            pose_results = pose_detector.detect(mp_image)

        # Debug: Log detection results
        pose_detected = pose_results and pose_results.pose_landmarks
        hands_detected = hands_results and hands_results.hand_landmarks

        logger.info(f"Pose detected: {pose_detected}, Hands detected: {hands_detected}")
        if pose_detected:
            logger.info(f"Pose landmarks count: {len(pose_results.pose_landmarks)}")
        if hands_detected:
            logger.info(f"Hand landmarks count: {len(hands_results.hand_landmarks)}")

        # Combine results
        prioritized_hands_list = []
        if hands_results and hands_results.hand_landmarks:
            prioritized_hands_list = prioritize_hands(hands_results, image_width, image_height)

        skeletons = process_combined_results(pose_results, hands_results, image_height, image_width, threshold, prioritized_hands_list)


        # Tracker Mode: Filter results if input shapes/states provided
        input_shapes = data.get('shapes') or data.get('states')
        if input_shapes and skeletons:
            # Interactive tracking: User wants to track a SPECIFIC object
            # We assume the first input shape is the target
            target_shape = input_shapes[0]
            target_center = get_shape_center(target_shape, image_width, image_height)

            if target_center:
                logger.info(f"Filtering {len(skeletons)} detections for target at {target_center}")

                # Find closest detected skeleton
                best_match = None
                min_dist = float('inf')

                for i, sk in enumerate(skeletons):
                    sk_center = get_shape_center(sk, image_width, image_height)
                    if not sk_center:
                        continue

                    dist = ((sk_center[0] - target_center[0])**2 + (sk_center[1] - target_center[1])**2)**0.5

                    # Log distance for debug
                    logger.info(f"Skeleton {i} distance: {dist:.1f}")

                    if dist < min_dist:
                        min_dist = dist
                        best_match = sk

                if best_match:
                    # Return only the best match
                    skeletons = [best_match]
                    logger.info(f"Selected match with distance {min_dist:.1f}")
                else:
                    skeletons = []

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
            "frame_number": "Frame number for timestamp calculation (video mode)",
            "tracking_mode": "'image' or 'video' mode for different detection strategies",
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
