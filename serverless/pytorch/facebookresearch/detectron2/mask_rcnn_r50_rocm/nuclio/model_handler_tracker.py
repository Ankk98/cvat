# Copyright (C) CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT

import numpy as np
import torch
from detectron2.model_zoo import get_config
from detectron2.data.detection_utils import convert_PIL_to_numpy
from detectron2.engine.defaults import DefaultPredictor
from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES
import cv2

CONFIG_OPTS = []
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.3  # For association

class ModelHandlerTracker:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        cfg = get_config('COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml')
        from detectron2 import model_zoo
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        cfg.MODEL.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = CONFIDENCE_THRESHOLD
        cfg.freeze()
        self.predictor = DefaultPredictor(cfg)

    def _calculate_iou(self, mask1, mask2):
        """Calculate IoU between two masks."""
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        return intersection / union if union > 0 else 0.0

    def _mask_to_polygon(self, mask, image_width, image_height):
        """Convert binary mask to polygon points."""
        if mask is None or not mask.any():
            return None

        # Get bounding box
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        if not rows.any() or not cols.any():
            return None

        # Find contours
        contours, _ = cv2.findContours(
            mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        if len(contours) == 0:
            return None

        # Get largest contour
        largest_contour = max(contours, key=cv2.contourArea)

        # Simplify polygon
        epsilon = 2.5
        approx_contour = cv2.approxPolyDP(largest_contour, epsilon, closed=True)

        # Convert to flat list [x1, y1, x2, y2, ...]
        polygon_points = []
        for point in approx_contour:
            polygon_points.extend([int(point[0][0]), int(point[0][1])])

        if len(polygon_points) < 6:
            return None

        return polygon_points

    def _run_detection(self, image):
        """Run Mask R-CNN detection on image."""
        image_np = convert_PIL_to_numpy(image, format="BGR")
        predictions = self.predictor(image_np)
        instances = predictions['instances']

        detections = []
        if instances.has("pred_masks"):
            for i in range(len(instances)):
                score = float(instances.scores[i])
                if score < CONFIDENCE_THRESHOLD:
                    continue

                label = COCO_CATEGORIES[int(instances.pred_classes[i])]["name"]
                mask = instances.pred_masks[i].cpu().numpy().astype(bool)

                detections.append({
                    'mask': mask,
                    'label': label,
                    'score': score,
                    'bbox': instances.pred_boxes[i].tensor[0].tolist()
                })

        return detections, image_np.shape[:2]

    def handle(self, image, shapes, states, frame, logger=None):
        """
        Track objects across frames using Mask R-CNN detection and IoU association.

        Args:
            image: PIL Image
            shapes: List of shapes (for initialization)
            states: List of tracking states
            frame: Current frame number
            logger: Logger instance

        Returns:
            Tuple of (new_shapes, new_states)
        """
        def log_info(msg):
            if logger:
                logger.info(msg)
            else:
                print(msg)

        def log_warning(msg):
            if logger:
                logger.warn(msg)
            else:
                print(f"WARNING: {msg}")

        log_info(f"[MASK_RCNN_TRACKER] handle called: frame={frame}, shapes_count={len(shapes) if shapes else 0}, states_count={len(states) if states else 0}")

        # Run detection
        detections, (image_height, image_width) = self._run_detection(image)
        log_info(f"[MASK_RCNN_TRACKER] Detected {len(detections)} objects")

        new_shapes = []
        new_states = []

        # If states are provided, continue tracking
        if states and len(states) > 0:
            log_info(f"[MASK_RCNN_TRACKER] Continuing tracking from {len(states)} states")

            # Create list of available detections
            available_detections = detections.copy()

            # For each state, find best matching detection
            for state_idx, state in enumerate(states):
                if state is None:
                    new_shapes.append(None)
                    new_states.append(None)
                    continue

                state_mask_data = state.get("mask", [])
                state_label = state.get("label", "")
                state_frame = state.get("frame", 0)

                if not state_mask_data:
                    log_warning(f"[MASK_RCNN_TRACKER] State {state_idx} has empty mask")
                    new_shapes.append(None)
                    new_states.append(None)
                    continue

                # Reconstruct state mask
                stored_height = state.get("image_height", image_height)
                stored_width = state.get("image_width", image_width)
                if isinstance(state_mask_data, list):
                    if len(state_mask_data) == stored_height * stored_width:
                        state_mask = np.array(state_mask_data, dtype=bool).reshape((stored_height, stored_width))
                    else:
                        state_mask = np.array(state_mask_data, dtype=bool)
                else:
                    state_mask = np.array(state_mask_data, dtype=bool)

                # Resize if necessary
                if state_mask.shape != (image_height, image_width):
                    state_mask = cv2.resize(
                        state_mask.astype(np.uint8),
                        (image_width, image_height),
                        interpolation=cv2.INTER_NEAREST
                    ).astype(bool)

                # Find best matching detection
                best_iou = -1
                best_detection_idx = -1

                for det_idx, detection in enumerate(available_detections):
                    if detection['label'] != state_label:
                        continue

                    iou = self._calculate_iou(state_mask, detection['mask'])
                    if iou > best_iou:
                        best_iou = iou
                        best_detection_idx = det_idx

                if best_iou >= IOU_THRESHOLD:
                    # Match found
                    detection = available_detections[best_detection_idx]
                    log_info(f"[MASK_RCNN_TRACKER] State {state_idx} matched to detection with IoU {best_iou:.3f}")

                    # Create polygon from mask
                    polygon_points = self._mask_to_polygon(detection['mask'], image_width, image_height)
                    if polygon_points:
                        new_shape = {
                            "type": "polygon",
                            "points": polygon_points,
                            "label": detection['label'],
                            "confidence": str(detection['score'])
                        }
                        new_shapes.append(new_shape)

                        # Update state
                        new_state = {
                            "mask": detection['mask'].astype(np.uint8).tolist(),
                            "label": detection['label'],
                            "frame": frame,
                            "image_height": image_height,
                            "image_width": image_width,
                        }
                        new_states.append(new_state)
                    else:
                        log_warning(f"[MASK_RCNN_TRACKER] Failed to create polygon for matched detection")
                        new_shapes.append(None)
                        new_states.append(None)

                    # Remove used detection
                    available_detections.pop(best_detection_idx)
                else:
                    # No match, keep state but no shape (gap)
                    log_info(f"[MASK_RCNN_TRACKER] State {state_idx} no match (best IoU {best_iou:.3f}), creating gap")
                    new_shapes.append(None)  # Gap in track
                    new_states.append(state)  # Keep state

        # If shapes provided (initialization) or no states, create new tracks from detections
        if (shapes and len(shapes) > 0) or not (states and len(states) > 0):
            if shapes and len(shapes) > 0:
                log_info(f"[MASK_RCNN_TRACKER] Initializing tracking from {len(shapes)} shapes")
                # For initialization from shapes, we could associate detections to shapes
                # But for simplicity, since shapes are provided, perhaps use them directly
                # But since this is for auto detection, we'll treat it as detection mode
            else:
                log_info(f"[MASK_RCNN_TRACKER] Creating new tracks from detections")

            for detection in detections:
                polygon_points = self._mask_to_polygon(detection['mask'], image_width, image_height)
                if polygon_points:
                    new_shape = {
                        "type": "polygon",
                        "points": polygon_points,
                        "label": detection['label'],
                        "confidence": str(detection['score'])
                    }
                    new_shapes.append(new_shape)

                    new_state = {
                        "mask": detection['mask'].astype(np.uint8).tolist(),
                        "label": detection['label'],
                        "frame": frame,
                        "image_height": image_height,
                        "image_width": image_width,
                    }
                    new_states.append(new_state)
                else:
                    log_warning(f"[MASK_RCNN_TRACKER] Failed to create polygon for detection")

        log_info(f"[MASK_RCNN_TRACKER] Completed: returning {len(new_shapes)} shapes, {len(new_states)} states")
        return new_shapes, new_states