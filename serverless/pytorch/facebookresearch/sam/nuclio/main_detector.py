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

        # Generate automatic masks
        masks = context.user_data.model.handle(image)

        context.logger.info(f"Generated {len(masks)} automatic masks")

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
