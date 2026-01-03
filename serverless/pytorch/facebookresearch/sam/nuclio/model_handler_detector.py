# Copyright (C) CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT

import numpy as np
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

class ModelHandlerDetector:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.sam_checkpoint = "/opt/nuclio/sam/sam_vit_h_4b8939.pth"
        self.model_type = "vit_h"

        # Load SAM model
        sam_model = sam_model_registry[self.model_type](checkpoint=self.sam_checkpoint)
        sam_model.to(device=self.device)

        # Create automatic mask generator
        # Optimized for speed: fewer points, higher thresholds, single crop layer
        self.mask_generator = SamAutomaticMaskGenerator(
            model=sam_model,
            points_per_side=16,  # Reduced from 32 for faster processing
            pred_iou_thresh=0.88,  # Higher threshold to filter more masks early
            stability_score_thresh=0.95,  # Higher threshold for stability
            crop_n_layers=0,  # No cropping for faster processing
            crop_n_points_downscale_factor=1,
            min_mask_region_area=500,  # Larger minimum area to reduce number of masks
        )

    def handle(self, image, threshold=0.5, max_masks=30, min_area=1000):
        """
        Generate automatic masks for the image with filtering.

        Args:
            image: PIL Image
            threshold: Minimum confidence (predicted_iou) threshold (0.0-1.0)
            max_masks: Maximum number of masks to return (sorted by confidence, default: 30)
            min_area: Minimum mask area in pixels (default: 1000 to filter small objects)
        """
        # Convert PIL to numpy array
        image_np = np.array(image)

        # Generate masks
        masks = self.mask_generator.generate(image_np)

        # Filter and sort masks by confidence
        filtered_masks = []
        for mask_data in masks:
            confidence = float(mask_data['predicted_iou'])
            area = int(mask_data['area'])

            # Apply filters
            if confidence < threshold:
                continue
            if area < min_area:
                continue

            filtered_masks.append({
                'mask_data': mask_data,
                'confidence': confidence,
                'area': area,
            })

        # Sort by confidence (descending) and take top N
        # This ensures only the best masks are returned, reducing false positives
        filtered_masks.sort(key=lambda x: x['confidence'], reverse=True)
        filtered_masks = filtered_masks[:max_masks]

        # Convert to CVAT format
        # CRITICAL: CVAT expects FLATTENED MASK PIXELS, not RLE!
        # Backend converts flattened pixels to RLE automatically (views.py:984-988)
        # Format: [pixel1, pixel2, ..., x_min, y_min, x_max, y_max]
        # This matches Mask R-CNN format (see mask_rcnn_r50_rocm/nuclio/main.py:117)
        cvat_masks = []
        for item in filtered_masks:
            mask_data = item['mask_data']
            # Get mask as binary array
            mask = mask_data['segmentation']

            # Convert to flattened pixels format (like Mask R-CNN)
            flattened_pixels = self.mask_to_flattened_pixels(mask)
            if not flattened_pixels:  # Skip empty masks
                continue

            cvat_mask = {
                "label": "object",
                "type": "mask",
                "confidence": str(item['confidence']),
                "mask": flattened_pixels,  # CVAT checks this first (views.py:1015)
                "points": flattened_pixels,  # CVAT copies mask to points, then processes it (views.py:1015, 1027-1031)
                "attributes": [
                    {"name": "confidence", "value": str(item['confidence'])},
                    {"name": "area", "value": str(item['area'])}
                ],
            }
            cvat_masks.append(cvat_mask)

        return cvat_masks

    def mask_to_flattened_pixels(self, mask):
        """
        Convert binary mask to CVAT flattened pixels format.

        CVAT expects: [pixel1, pixel2, ..., x_min, y_min, x_max, y_max]
        - Flattened pixels are 0/1 values (not RLE)
        - Backend converts to RLE automatically (views.py:984-988)
        - Matches Mask R-CNN format (mask_rcnn_r50_rocm/nuclio/main.py:117)
        """
        # Get bounding box
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)

        if not np.any(rows) or not np.any(cols):
            # Empty mask
            return []

        # Get tight bounding box coordinates
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]

        # Extract tight mask and flatten it (row-major order)
        tight_mask = mask[y_min:y_max+1, x_min:x_max+1]
        flattened_pixels = tight_mask.flatten().astype(np.uint8).tolist()

        # Convert to 0/1 (boolean to int) - ensure binary values
        flattened_pixels = [int(x) for x in flattened_pixels]

        # Append bbox coordinates [x_min, y_min, x_max, y_max]
        # Frontend expects inclusive end coordinates
        cvat_mask = flattened_pixels + [int(x_min), int(y_min), int(x_max), int(y_max)]

        # Validate minimum length (frontend requires >= 6)
        if len(cvat_mask) < 6:
            return []

        return cvat_mask
