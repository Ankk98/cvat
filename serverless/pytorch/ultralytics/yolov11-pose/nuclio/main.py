# Copyright (C) CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT

import json
import base64
import io
from PIL import Image
from ultralytics import YOLO
import numpy as np

def init_context(context):
    context.logger.info("Initializing YOLO11 Pose model...")

    # Load YOLO11 pose model
    model = YOLO('yolo11n-pose.pt')  # Use nano model for efficiency

    context.user_data.model = model
    context.logger.info("YOLO11 Pose model initialized successfully")

def handler(context, event):
    context.logger.info("Running YOLO11 Pose estimation")
    data = event.body
    buf = io.BytesIO(base64.b64decode(data["image"]))
    threshold = float(data.get("threshold", 0.5))
    image = Image.open(buf).convert("RGB")

    # Run inference
    results = context.user_data.model(np.array(image), conf=threshold)

    output_results = []

    # Process results
    for result in results:
        if result.keypoints is not None:
            keypoints = result.keypoints.xy.cpu().numpy()  # [num_persons, 17, 2]
            confidences = result.keypoints.conf.cpu().numpy() if result.keypoints.conf is not None else np.ones((keypoints.shape[0], keypoints.shape[1]))

            for person_idx in range(keypoints.shape[0]):
                person_keypoints = keypoints[person_idx]
                person_confidences = confidences[person_idx]

                # Create skeleton for this person
                skeleton = {
                    "confidence": str(result.boxes.conf[person_idx].item()) if result.boxes is not None else "1.0",
                    "label": "person",
                    "type": "skeleton",
                    "elements": []
                }

                # YOLO11 pose keypoint order: [nose, left_eye, right_eye, left_ear, right_ear, left_shoulder, right_shoulder,
                #                               left_elbow, right_elbow, left_wrist, right_wrist, left_hip, right_hip,
                #                               left_knee, right_knee, left_ankle, right_ankle]
                keypoint_names = [
                    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
                    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                    "left_wrist", "right_wrist", "left_hip", "right_hip",
                    "left_knee", "right_knee", "left_ankle", "right_ankle"
                ]

                for kp_idx, (x, y) in enumerate(person_keypoints):
                    confidence = float(person_confidences[kp_idx]) if kp_idx < len(person_confidences) else 0.0

                    element = {
                        "label": keypoint_names[kp_idx],
                        "type": "points",
                        "outside": 0 if confidence > threshold else 1,
                        "points": [float(x), float(y)],
                        "confidence": str(confidence)
                    }
                    skeleton["elements"].append(element)

                # Only add skeleton if it has visible keypoints
                if not all([element['outside'] for element in skeleton["elements"]]):
                    output_results.append(skeleton)

    return context.Response(
        body=json.dumps(output_results),
        headers={},
        content_type='application/json',
        status_code=200
    )
