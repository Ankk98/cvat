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
        self.mask_generator = SamAutomaticMaskGenerator(
            model=sam_model,
            points_per_side=32,  # Balance between quality and speed
            pred_iou_thresh=0.86,  # Filter low-quality masks
            stability_score_thresh=0.92,  # Filter unstable masks
            crop_n_layers=1,  # Single layer cropping
            crop_n_points_downscale_factor=2,  # Downscale factor for points
            min_mask_region_area=100,  # Minimum mask area
        )

    def handle(self, image):
        """Generate automatic masks for the image."""
        # Convert PIL to numpy array
        image_np = np.array(image)

        # Generate masks
        masks = self.mask_generator.generate(image_np)

        # Convert to CVAT format
        cvat_masks = []
        for i, mask_data in enumerate(masks):
            # Get mask as binary array
            mask = mask_data['segmentation']

            # Convert to RLE (Run Length Encoding) for CVAT
            rle = self.mask_to_rle(mask)

            cvat_mask = {
                "label": "object",
                "type": "mask",
                "confidence": float(mask_data['predicted_iou']),
                "attributes": [
                    {"name": "confidence", "value": str(mask_data['predicted_iou'])},
                    {"name": "area", "value": str(mask_data['area'])}
                ],
                "mask": rle
            }
            cvat_masks.append(cvat_mask)

        return cvat_masks

    def mask_to_rle(self, mask):
        """Convert binary mask to RLE format for CVAT."""
        # Flatten the mask
        mask_flat = mask.flatten()

        # Find runs of True values
        runs = []
        current_run = 0

        for i, val in enumerate(mask_flat):
            if val:  # True/1
                current_run += 1
            else:  # False/0
                if current_run > 0:
                    runs.append(current_run)
                    current_run = 0
                runs.append(0)

        # Handle final run if mask ends with True
        if current_run > 0:
            runs.append(current_run)

        return runs
