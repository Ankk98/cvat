import base64
import json

from model_handler import PointCloudDetector


def init_context(context):
    context.logger.info("Initializing SIT point cloud detector (ROCm)")
    context.user_data.detector = PointCloudDetector(context.logger)
    context.logger.info("SIT point cloud detector ready")


def handler(context, event):
    data = event.body
    frame_id = data.get("frame", -1)
    threshold = data.get("threshold")

    cloud_bytes = base64.b64decode(data["image"])

    detections = context.user_data.detector.infer(
        cloud_bytes=cloud_bytes,
        threshold=threshold,
        frame_id=frame_id,
    )

    return context.Response(
        body=json.dumps(detections),
        headers={},
        content_type="application/json",
        status_code=200,
    )

