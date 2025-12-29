"""
Mask R-CNN R50 Instance Segmentation for CVAT
==============================================

Clean implementation of Mask R-CNN R50 for instance segmentation.
Outputs masks in CVAT-compatible flattened pixels format.

Model: COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x
Output: Instance segmentation masks (flattened pixels, converted to RLE by CVAT backend)

Format: [pixel1, pixel2, ..., x_min, y_min, x_max, y_max]
- Matches OpenVINO to_cvat_mask() format
- CVAT backend converts flattened pixels to RLE automatically
"""

import json
import base64
import io
from PIL import Image
import numpy as np

import torch
from detectron2.model_zoo import get_config
from detectron2.data.detection_utils import convert_PIL_to_numpy
from detectron2.engine.defaults import DefaultPredictor
from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES

CONFIG_OPTS = []
CONFIDENCE_THRESHOLD = 0.5


def init_context(context):
    """Initialize Mask R-CNN model."""
    context.logger.info("Init context...  0%")

    # Use Mask R-CNN for instance segmentation (returns masks)
    cfg = get_config('COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml')

    # Use model zoo weights (will download automatically)
    from detectron2 import model_zoo
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

    # Configure device
    if torch.cuda.is_available():
        cfg.MODEL.DEVICE = 'cuda'
        context.logger.info("Using CUDA device")
    else:
        cfg.MODEL.DEVICE = 'cpu'
        context.logger.info("Using CPU device")

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = CONFIDENCE_THRESHOLD
    cfg.freeze()
    predictor = DefaultPredictor(cfg)

    context.user_data.model_handler = predictor

    context.logger.info("Init context...100%")


def handler(context, event):
    """Handle inference request."""
    context.logger.info("Run mask_rcnn-R50 model (instance segmentation)")
    data = event.body
    buf = io.BytesIO(base64.b64decode(data["image"]))
    threshold = float(data.get("threshold", 0.5))
    image = convert_PIL_to_numpy(Image.open(buf), format="BGR")
    height, width = image.shape[:2]

    predictions = context.user_data.model_handler(image)

    instances = predictions['instances']
    results = []

    # Check if masks are available (Mask R-CNN should always have them)
    has_masks = instances.has("pred_masks")
    if not has_masks:
        context.logger.warning("⚠️  Mask R-CNN should have masks, but has_masks is False!")

    for i in range(len(instances)):
        score = float(instances.scores[i])
        if score < threshold:
            continue

        label = COCO_CATEGORIES[int(instances.pred_classes[i])]["name"]
        box = instances.pred_boxes[i].tensor[0].tolist()  # [x1, y1, x2, y2]
        # Convert NumPy float types to native Python types for JSON serialization
        box = [float(x) for x in box]

        # Mask R-CNN always outputs masks, so type should always be "mask"
        if has_masks:
            mask = instances.pred_masks[i].cpu().numpy().astype(np.uint8)

            # CRITICAL: CVAT backend expects FLATTENED MASK PIXELS, not RLE!
            # Backend at line 1029 calls mask_tools.mask_to_rle() on cut_points
            # This function expects a 2D mask array, so we provide flattened pixels
            # Format: [pixel1, pixel2, ..., x_min, y_min, x_max, y_max]
            # Reference: serverless/openvino/base/shared.py:to_cvat_mask()

            # Get tight bounding box
            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)
            if not rows.any() or not cols.any():
                context.logger.warning(f"⚠️  Empty mask for detection {i}, skipping")
                continue

            y_min, y_max = np.where(rows)[0][[0, -1]]
            x_min, x_max = np.where(cols)[0][[0, -1]]

            # Extract tight mask and flatten it (like to_cvat_mask does)
            tight_mask = mask[y_min:y_max+1, x_min:x_max+1]
            flattened_pixels = tight_mask.flat[:].tolist()  # Flatten to 1D list
            # Convert to 0/1 (boolean to int)
            flattened_pixels = [int(x) for x in flattened_pixels]

            # Append bbox coordinates [x_min, y_min, x_max, y_max]
            # Frontend expects inclusive end coordinates (shared.ts:453 uses right-left+1)
            cvat_mask = flattened_pixels + [int(x_min), int(y_min), int(x_max), int(y_max)]

            # Validate minimum length (frontend requires >= 6)
            if len(cvat_mask) < 6:
                context.logger.warning(f"⚠️  Invalid mask format for detection {i}, skipping")
                continue

            result = {
                "confidence": str(score),
                "label": label,
                "mask": cvat_mask,  # CVAT checks this first (views.py:1015)
                "points": cvat_mask,  # CVAT copies mask to points, then processes it (views.py:1015, 1027-1031)
                "type": "mask",
            }
        else:
            # Fallback to bounding box if masks are somehow unavailable
            context.logger.warning(f"⚠️  No mask available for detection {i}, using bounding box")
            result = {
                "confidence": str(score),
                "label": label,
                "points": box,
                "type": "rectangle",
            }

        results.append(result)

    context.logger.info(f"Detected {len(results)} instances with masks")
    return context.Response(body=json.dumps(results), headers={},
        content_type='application/json', status_code=200)


# NOTE: _mask_to_rle() function removed - we now provide flattened pixels instead
# CVAT backend expects flattened mask pixels (not RLE), then converts to RLE internally
# This matches the format used by OpenVINO Mask R-CNN (to_cvat_mask function)

