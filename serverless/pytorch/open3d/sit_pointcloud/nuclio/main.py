import base64
import json

from model_handler import PointCloudDetector


def init_context(context):
    """
    Initialize the SIT Point Cloud Detector when Nuclio function starts.

    This creates a single detector instance that persists across requests,
    allowing parameter tuning and GPU memory reuse for better performance.
    """
    context.logger.info("Initializing SIT point cloud detector (ROCm)")
    context.user_data.detector = PointCloudDetector(context.logger)
    context.logger.info("SIT point cloud detector ready")


def handler(context, event):
    """
    Nuclio handler for processing point cloud detection requests.

    CVAT sends requests with:
    - "image": base64-encoded point cloud data (.pcd format after CVAT conversion)
    - "frame": frame number for logging
    - "threshold": optional confidence threshold override

    Returns CVAT-compatible cuboid detections with position, size, and heading.
    """
    # Parse CVAT request data
    data = event.body
    frame_id = data.get("frame", -1)  # Frame number for logging
    threshold = data.get("threshold")  # Optional confidence threshold

    # Decode base64 point cloud data
    # CVAT sends .pcd format (converted from original .bin files)
    cloud_bytes = base64.b64decode(data["image"])

    # Run detection on the point cloud
    detections = context.user_data.detector.infer(
        cloud_bytes=cloud_bytes,
        threshold=threshold,
        frame_id=frame_id,
    )

    # Return detections in CVAT-compatible JSON format
    return context.Response(
        body=json.dumps(detections),
        headers={},
        content_type="application/json",
        status_code=200,
    )

