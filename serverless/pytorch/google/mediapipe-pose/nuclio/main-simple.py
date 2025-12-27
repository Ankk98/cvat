# Copyright (C) CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT

import json
import base64
import io
import mediapipe as mp
import cv2
import numpy as np
from PIL import Image

def init_context(context):
    context.logger.info("Initializing MediaPipe Pose (Simple)...")

    # Initialize MediaPipe Pose with minimal config
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=True,
        model_complexity=0,  # Lowest complexity for speed
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    context.user_data.pose = pose
    context.user_data.mp_pose = mp_pose
    context.logger.info("MediaPipe Pose (Simple) initialized successfully")

def handler(context, event):
    context.logger.info("Running MediaPipe Pose (Simple)")
    data = event.body
    buf = io.BytesIO(base64.b64decode(data["image"]))
    threshold = float(data.get("threshold", 0.3))  # Lower threshold for simpler model
    image = Image.open(buf).convert("RGB")

    # Convert PIL to numpy array
    image_np = np.array(image)

    # Convert RGB to BGR for MediaPipe
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # Run pose estimation
    results = context.user_data.pose.process(image_bgr)

    output_results = []

    if results.pose_landmarks:
        # MediaPipe provides landmarks with x, y, z coordinates (normalized 0-1)
        # and visibility scores
        landmarks = results.pose_landmarks.landmark

        # Convert to CVAT skeleton format (simplified keypoints)
        skeleton = {
            "confidence": str(results.pose_landmarks.landmark[0].visibility),  # Use nose visibility
            "label": "pose",
            "type": "skeleton",
            "elements": []
        }

        # Simplified keypoint mapping (similar to COCO)
        # MediaPipe indices to simplified labels
        keypoint_map = {
            0: "nose",
            11: "left_shoulder",
            12: "right_shoulder",
            13: "left_elbow",
            14: "right_elbow",
            15: "left_wrist",
            16: "right_wrist",
            23: "left_hip",
            24: "right_hip",
            25: "left_knee",
            26: "right_knee"
        }

        # Add each landmark as an element
        for mp_idx, label in keypoint_map.items():
            landmark = landmarks[mp_idx]

            # MediaPipe coordinates are normalized (0-1), convert to pixel coordinates
            x = int(landmark.x * image.width)
            y = int(landmark.y * image.height)

            element = {
                "label": label,
                "type": "points",
                "outside": 0 if landmark.visibility > threshold else 1,
                "points": [x, y],
                "confidence": str(landmark.visibility)
            }
            skeleton["elements"].append(element)

        # Only add skeleton if it has visible keypoints
        if not all([element['outside'] for element in skeleton["elements"]]):
            output_results.append(skeleton)

    context.logger.info(f"Detected {len(output_results)} poses")
    return context.Response(
        body=json.dumps(output_results),
        headers={},
        content_type='application/json',
        status_code=200
    )
