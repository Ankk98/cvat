# Copyright (C) CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT

import json
import base64
from PIL import Image
import io
from model_handler_tracker import ModelHandlerTracker

def init_context(context):
    context.logger.info("Init SAM Tracker context...  0%")
    model = ModelHandlerTracker()
    context.user_data.model = model
    context.logger.info("Init SAM Tracker context...100%")

def handler(context, event):
    context.logger.info("[SAM_TRACKER_MAIN] Handler called")
    data = event.body

    # Decode image
    buf = io.BytesIO(base64.b64decode(data["image"]))
    image = Image.open(buf)
    image = image.convert("RGB")
    context.logger.info(f"[SAM_TRACKER_MAIN] Image decoded: size={image.size}, mode={image.mode}")

    # Get shapes and states
    shapes = data.get("shapes", [])
    states = data.get("states", [])
    frame = data.get("frame", 0)
    # CRITICAL: If states are provided, it's continuation (track), not initialization
    # Only use init_tracking if shapes are provided AND no states
    request_type = "track" if states else ("init_tracking" if shapes else "track")

    context.logger.info(f"[SAM_TRACKER_MAIN] Request type: {request_type}, frame={frame}, shapes_count={len(shapes)}, states_count={len(states)}")

    try:
        # Track shapes (pass logger to handle method)
        new_shapes, new_states = context.user_data.model.handle(
            image, shapes, states, frame, logger=context.logger
        )

        # Format response for CVAT tracker
        response = {
            "shapes": new_shapes,
            "states": new_states,
        }

        successful_shapes = len([s for s in new_shapes if s is not None])
        context.logger.info(f"[SAM_TRACKER_MAIN] Tracking completed: {successful_shapes}/{len(new_shapes)} successful shapes, {len(new_states)} states")

        return context.Response(
            body=json.dumps(response),
            headers={},
            content_type='application/json',
            status_code=200
        )
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        context.logger.error(f"[SAM_TRACKER_MAIN] Error in SAM tracker: {e}\n{error_traceback}")
        return context.Response(
            body=json.dumps({"error": str(e)}),
            headers={},
            content_type='application/json',
            status_code=500
        )

