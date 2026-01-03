# Copyright (C) CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT

import json
import base64
from PIL import Image
import io
from model_handler_detector import ModelHandlerDetector

def init_context(context):
    context.logger.info("Init SAM Auto Segmentation context...  0%")
    model = ModelHandlerDetector()
    context.user_data.model = model
    context.logger.info("Init SAM Auto Segmentation context...100%")

def handler(context, event):
    context.logger.info("SAM Auto Segmentation handler called")

    try:
        data = event.body
        buf = io.BytesIO(base64.b64decode(data["image"]))
        image = Image.open(buf)
        image = image.convert("RGB")

        # Extract filtering parameters
        # threshold: from CVAT UI (user can adjust)
        # max_masks: limit number of masks to avoid too many false positives (default: 30)
        # min_area: filter out very small masks (default: 1000 pixels)
        threshold = float(data.get("threshold", 0.5))  # Confidence threshold from UI
        max_masks = int(data.get("max_masks", 30))  # Maximum number of masks (top N by confidence)
        min_area = int(data.get("min_area", 1000))  # Minimum mask area in pixels

        # Generate automatic masks with filtering
        masks = context.user_data.model.handle(image, threshold=threshold, max_masks=max_masks, min_area=min_area)

        context.logger.info(f"Generated {len(masks)} automatic masks (after filtering)")

        return context.Response(
            body=json.dumps(masks),
            headers={},
            content_type='application/json',
            status_code=200
        )

    except Exception as e:
        context.logger.error(f"Error in SAM auto segmentation: {e}")
        return context.Response(
            body=json.dumps({"error": str(e)}),
            headers={},
            content_type='application/json',
            status_code=500
        )
