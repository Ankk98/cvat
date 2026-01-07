# Copyright (C) CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT

import numpy as np
import torch
import cv2
from PIL import Image
from segment_anything import sam_model_registry, SamPredictor
# cvat_sdk.masks will be imported conditionally to handle cases where it's not available

class ModelHandlerTracker:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.sam_checkpoint = "/opt/nuclio/sam/sam_vit_h_4b8939.pth"
        self.model_type = "vit_h"
        sam_model = sam_model_registry[self.model_type](checkpoint=self.sam_checkpoint)
        sam_model.to(device=self.device)
        self.predictor = SamPredictor(sam_model)
        self.current_image_embedding = None
        self.current_image = None

    def _shape_to_mask(self, shape, image_width, image_height):
        """Convert CVAT shape (mask or polygon) to binary mask array."""
        if shape["type"] == "mask":
            # Decode mask from RLE format using cvat_sdk
            try:
                from cvat_sdk.masks import decode_mask
                points = shape["points"]
                if len(points) < 5:
                    return None
                mask = decode_mask(points, image_width=image_width, image_height=image_height)
                return mask
            except ImportError:
                # Fallback: manual RLE decoding
                points = shape["points"]
                if len(points) < 5:
                    return None

                # Extract bounding box
                x_min, y_min, x_max, y_max = [int(x) for x in points[-4:]]
                rle_counts = points[:-4]

                # Decode RLE to mask using CVAT format
                # CVAT RLE: alternating run lengths starting with background (0) or foreground (1)
                mask = np.zeros((image_height, image_width), dtype=np.uint8)
                if len(rle_counts) > 0:
                    tight_width = x_max - x_min + 1
                    tight_height = y_max - y_min + 1
                    total_pixels = tight_width * tight_height

                    # Build flat mask from RLE
                    flat_mask = np.zeros(total_pixels, dtype=bool)
                    pos = 0
                    is_foreground = False  # Start with background

                    for length in rle_counts:
                        length = int(length)
                        if pos + length <= total_pixels:
                            if is_foreground:
                                flat_mask[pos:pos+length] = True
                            pos += length
                            is_foreground = not is_foreground
                        else:
                            break

                    # Reshape and place in full image
                    tight_mask = flat_mask.reshape((tight_height, tight_width))
                    mask[y_min:y_max+1, x_min:x_max+1] = tight_mask.astype(np.uint8)

                return mask.astype(bool)

        elif shape["type"] == "polygon":
            # Convert polygon points to mask
            points = shape["points"]
            if len(points) < 6 or len(points) % 2 != 0:
                return None

            # Reshape points to [x, y] pairs
            points_array = np.array(points, dtype=np.int32).reshape((-1, 2))

            # Create mask
            mask = np.zeros((image_height, image_width), dtype=np.uint8)
            cv2.fillPoly(mask, [points_array], 1)
            return mask.astype(bool)

        return None

    def _mask_to_shape(self, mask, shape_type, image_width, image_height):
        """Convert binary mask to CVAT shape format."""
        if mask is None or not mask.any():
            return None

        # Get bounding box
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        if not rows.any() or not cols.any():
            return None

        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]

        if shape_type == "mask":
            # Extract tight mask
            tight_mask = mask[y_min:y_max+1, x_min:x_max+1]

            # Encode to RLE format using cvat_sdk
            try:
                from cvat_sdk.masks import encode_mask
                encoded = encode_mask(tight_mask, bbox=[x_min, y_min, x_max+1, y_max+1])
            except ImportError:
                # Fallback: simple RLE encoding (CVAT format)
                flat_mask = tight_mask.ravel().astype(bool)
                rle = []

                if len(flat_mask) == 0:
                    encoded = [0, x_min, y_min, x_max, y_max]
                else:
                    # CVAT RLE: alternating run lengths, starting with background
                    current_val = False
                    run_length = 0

                    for val in flat_mask:
                        if val == current_val:
                            run_length += 1
                        else:
                            if run_length > 0:
                                rle.append(run_length)
                            current_val = val
                            run_length = 1

                    if run_length > 0:
                        rle.append(run_length)

                    # Ensure we start with background (if first pixel is foreground, prepend 0)
                    if len(rle) > 0 and flat_mask[0]:
                        rle.insert(0, 0)

                    # Add bbox coordinates
                    encoded = rle + [x_min, y_min, x_max, y_max]

            return {
                "type": "mask",
                "points": encoded,
            }
        elif shape_type == "polygon":
            # Find contours and get largest
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

            return {
                "type": "polygon",
                "points": polygon_points,
            }

        return None

    def handle(self, image, shapes, states, frame, logger=None):
        """
        Track shapes across frames using SAM.

        Args:
            image: PIL Image
            shapes: List of shapes to track (for initialization) or None (for continuation)
            states: List of tracking states (for continuation) or empty (for initialization)
            frame: Current frame number
            logger: Optional logger instance (from context.logger)

        Returns:
            Tuple of (new_shapes, new_states)
        """
        # Use provided logger or print as fallback
        def log_info(msg):
            if logger:
                logger.info(msg)
            else:
                print(msg)

        def log_warning(msg):
            if logger:
                # Nuclio logger uses 'warn' instead of 'warning'
                if hasattr(logger, 'warn'):
                    logger.warn(msg)
                elif hasattr(logger, 'warning'):
                    logger.warning(msg)
                else:
                    logger.error(f"WARNING: {msg}")
            else:
                print(f"WARNING: {msg}")

        log_info(f"[SAM_TRACKER] handle called: frame={frame}, shapes_count={len(shapes) if shapes else 0}, states_count={len(states) if states else 0}")

        image_np = np.array(image.convert("RGB"))
        image_height, image_width = image_np.shape[:2]
        log_info(f"[SAM_TRACKER] Image dimensions: {image_width}x{image_height}")

        # Set image embedding (only if image changed)
        if self.current_image is None or not np.array_equal(np.array(self.current_image), image_np):
            self.predictor.set_image(image_np)
            self.current_image = image_np
            self.current_image_embedding = self.predictor.get_image_embedding()

        new_shapes = []
        new_states = []

        # CRITICAL: Prioritize states (continuation) over shapes (initialization)
        # If states are provided, use continuation even if shapes are also provided
        # Initialize tracking from shapes (only if no states)
        if (shapes and len(shapes) > 0) and not (states and len(states) > 0):
            log_info(f"[SAM_TRACKER] Initializing tracking from {len(shapes)} shape(s)")
            for idx, shape in enumerate(shapes):
                if shape is None:
                    log_warning(f"[SAM_TRACKER] Shape {idx} is None, skipping")
                    continue

                log_info(f"[SAM_TRACKER] Processing shape {idx}: type={shape.get('type')}, label={shape.get('label')}")

                # Convert shape to mask
                mask = self._shape_to_mask(shape, image_width, image_height)
                if mask is None:
                    log_warning(f"[SAM_TRACKER] Failed to convert shape {idx} to mask, skipping")
                    continue

                log_info(f"[SAM_TRACKER] Shape {idx} converted to mask: shape={mask.shape}, sum={mask.sum()}")

                # Use mask to get center point for SAM predictor
                # For initialization, we use point prompts only (not mask_input)
                # mask_input is for refinement/continuation, not initialization
                center_y, center_x = np.unravel_index(np.argmax(mask), mask.shape)
                log_info(f"[SAM_TRACKER] Shape {idx} center point: ({center_x}, {center_y})")

                # Set image for predictor
                self.predictor.set_image(image_np)
                log_info(f"[SAM_TRACKER] Set image for predictor (frame {frame})")

                # For initialization, use point prompts only (no mask_input)
                # This avoids dimension mismatch issues
                # We'll use multiple points from the mask to get better results
                # Get a few points from the mask (center and a few others)
                points = []
                labels = []

                # Add center point as positive
                points.append([center_x, center_y])
                labels.append(1)

                # Add a few more points from the mask for better initialization
                y_coords, x_coords = np.where(mask)
                if len(y_coords) > 0:
                    # Add top-left, top-right, bottom-left, bottom-right points
                    if len(y_coords) > 4:
                        y_min, y_max = y_coords.min(), y_coords.max()
                        x_min, x_max = x_coords.min(), x_coords.max()
                        points.extend([
                            [x_min, y_min],  # top-left
                            [x_max, y_min],  # top-right
                            [x_min, y_max],  # bottom-left
                            [x_max, y_max],  # bottom-right
                        ])
                        labels.extend([1, 1, 1, 1])
                    else:
                        # If mask is small, just use all points
                        for i in range(min(4, len(y_coords))):
                            points.append([x_coords[i], y_coords[i]])
                            labels.append(1)

                log_info(f"[SAM_TRACKER] Shape {idx} using {len(points)} point(s) for initialization")

                # Predict mask using SAM with point prompts only (no mask_input for initialization)
                masks, scores, logits = self.predictor.predict(
                    point_coords=np.array(points),
                    point_labels=np.array(labels),
                    multimask_output=True,  # Get multiple masks to choose the best one
                )

                # Select the best mask
                # For initialization, we can use the highest score since we don't have a previous mask to compare
                best_idx = np.argmax(scores)
                log_info(f"[SAM_TRACKER] Shape {idx} selected mask {best_idx} with score {scores[best_idx]:.4f} from {len(scores)} masks")

                # Get best mask (use the selected index)
                best_mask = masks[best_idx] > 0.0
                log_info(f"[SAM_TRACKER] Shape {idx} best mask: shape={best_mask.shape}, sum={best_mask.sum()}")

                # Convert back to shape
                # CRITICAL: If original shape was mask but we're creating a polygon track,
                # we should return polygon. However, we don't know the track type here.
                # The safest approach is to preserve the original type, but the frontend
                # will handle conversion if needed (see tools-control.tsx)
                shape_type = shape.get("type", "mask")
                new_shape = self._mask_to_shape(best_mask, shape_type, image_width, image_height)

                if new_shape:
                    new_shape["label"] = shape.get("label", "object")
                    log_info(f"[SAM_TRACKER] Shape {idx} converted to output: type={new_shape.get('type')}, points_count={len(new_shape.get('points', []))}")
                    new_shapes.append(new_shape)

                    # Store state (mask for next frame - embedding is not needed as we set_image each frame)
                    # Store mask as 2D list for easier reconstruction
                    mask_list = best_mask.astype(np.uint8).tolist()
                    state = {
                        "mask": mask_list,
                        "frame": frame,
                        "shape_type": shape_type,
                        "label": new_shape["label"],
                        "image_height": image_height,
                        "image_width": image_width,
                    }
                    new_states.append(state)
                    log_info(f"[SAM_TRACKER] Shape {idx} state stored: frame={frame}, mask_shape={len(mask_list)}x{len(mask_list[0]) if mask_list else 0}")
                else:
                    log_warning(f"[SAM_TRACKER] Shape {idx} failed to convert mask to shape")
                    new_shapes.append(None)
                    new_states.append(None)

        # Continue tracking from states
        elif states and len(states) > 0:
            log_info(f"[SAM_TRACKER] Continuing tracking from {len(states)} state(s)")
            for idx, state in enumerate(states):
                if state is None:
                    log_warning(f"[SAM_TRACKER] State {idx} is None, skipping")
                    new_shapes.append(None)
                    new_states.append(None)
                    continue

                log_info(f"[SAM_TRACKER] Processing state {idx}: frame={state.get('frame')}, shape_type={state.get('shape_type')}, label={state.get('label')}")

                # Reconstruct mask from state
                mask_data = state.get("mask", [])
                if not mask_data:
                    log_warning(f"[SAM_TRACKER] State {idx} has empty mask, skipping")
                    new_shapes.append(None)
                    new_states.append(None)
                    continue

                # Handle mask reconstruction from state
                # State stores mask as 2D list with dimensions stored
                stored_height = state.get("image_height", image_height)
                stored_width = state.get("image_width", image_width)

                if isinstance(mask_data, list):
                    if isinstance(mask_data[0], list):
                        # 2D list - direct conversion
                        prev_mask = np.array(mask_data, dtype=np.uint8).astype(bool)
                    else:
                        # Flat list - reshape using stored dimensions
                        if len(mask_data) == stored_height * stored_width:
                            prev_mask = np.array(mask_data, dtype=np.uint8).reshape((stored_height, stored_width)).astype(bool)
                        elif len(mask_data) == image_height * image_width:
                            # Try current dimensions as fallback
                            prev_mask = np.array(mask_data, dtype=np.uint8).reshape((image_height, image_width)).astype(bool)
                        else:
                            new_shapes.append(None)
                            new_states.append(None)
                            continue
                else:
                    prev_mask = np.array(mask_data, dtype=np.uint8).astype(bool)

                log_info(f"[SAM_TRACKER] State {idx} reconstructed mask: shape={prev_mask.shape}, sum={prev_mask.sum()}")

                if prev_mask.size == 0:
                    log_warning(f"[SAM_TRACKER] State {idx} reconstructed mask is empty, skipping")
                    new_shapes.append(None)
                    new_states.append(None)
                    continue

                # Resize mask if image size changed
                if prev_mask.shape != (image_height, image_width):
                    log_info(f"[SAM_TRACKER] State {idx} resizing mask from {prev_mask.shape} to ({image_height}, {image_width})")
                    prev_mask = cv2.resize(
                        prev_mask.astype(np.uint8),
                        (image_width, image_height),
                        interpolation=cv2.INTER_NEAREST
                    ).astype(bool)

                # Set image for predictor
                if self.current_image is None or not np.array_equal(np.array(self.current_image), image_np):
                    self.predictor.set_image(image_np)
                    self.current_image = image_np
                    self.current_image_embedding = self.predictor.get_image_embedding()
                    log_info(f"[SAM_TRACKER] Set image for predictor (frame {frame})")

                # Get multiple points from the previous mask for better tracking
                # Use a grid of points distributed across the mask for more accurate tracking
                y_coords, x_coords = np.where(prev_mask)

                if len(y_coords) == 0:
                    log_warning(f"[SAM_TRACKER] State {idx} has no mask pixels, skipping")
                    new_shapes.append(None)
                    new_states.append(None)
                    continue

                points = []
                labels = []

                # Strategy: Use a combination of center, corners, and distributed points
                # This provides better spatial coverage for tracking

                # 1. Add center point (most reliable)
                center_y, center_x = np.unravel_index(np.argmax(prev_mask), prev_mask.shape)
                points.append([center_x, center_y])
                labels.append(1)
                log_info(f"[SAM_TRACKER] State {idx} center point: ({center_x}, {center_y})")

                # 2. Add corner points (bounding box corners)
                y_min, y_max = y_coords.min(), y_coords.max()
                x_min, x_max = x_coords.min(), x_coords.max()
                points.extend([
                    [x_min, y_min],  # top-left
                    [x_max, y_min],  # top-right
                    [x_min, y_max],  # bottom-left
                    [x_max, y_max],  # bottom-right
                ])
                labels.extend([1, 1, 1, 1])

                # 3. Add distributed points across the mask (grid sampling)
                # Sample points evenly across the mask area for better coverage
                num_samples = min(10, len(y_coords))  # Use up to 10 additional points
                if num_samples > 0:
                    # Sample evenly spaced indices
                    indices = np.linspace(0, len(y_coords) - 1, num_samples, dtype=int)
                    for idx in indices:
                        points.append([x_coords[idx], y_coords[idx]])
                        labels.append(1)

                log_info(f"[SAM_TRACKER] State {idx} using {len(points)} point(s) for tracking continuation (center + corners + {num_samples} distributed points)")

                # Predict new mask using point prompts (avoiding mask_input dimension issues)
                masks, scores, logits = self.predictor.predict(
                    point_coords=np.array(points),
                    point_labels=np.array(labels),
                    multimask_output=True,  # Get multiple masks to choose the best one
                )

                # CRITICAL: Select the mask that best matches the previous mask (for tracking continuity)
                # Use multiple criteria: IoU, center distance, and SAM confidence score

                # Calculate previous mask center for distance comparison
                prev_y_coords, prev_x_coords = np.where(prev_mask)
                if len(prev_y_coords) > 0:
                    prev_center_y = prev_y_coords.mean()
                    prev_center_x = prev_x_coords.mean()
                else:
                    prev_center_y, prev_center_x = 0, 0

                # Evaluate all masks
                mask_candidates = []

                for mask_idx, pred_mask in enumerate(masks):
                    pred_mask_bool = pred_mask > 0.0

                    # 1. Calculate IoU (Intersection over Union) with previous mask
                    intersection = np.logical_and(pred_mask_bool, prev_mask).sum()
                    union = np.logical_or(pred_mask_bool, prev_mask).sum()
                    if union > 0:
                        iou = intersection / union
                    else:
                        iou = 0.0

                    # 2. Calculate center distance (normalized by image diagonal)
                    pred_y_coords, pred_x_coords = np.where(pred_mask_bool)
                    if len(pred_y_coords) > 0:
                        pred_center_y = pred_y_coords.mean()
                        pred_center_x = pred_x_coords.mean()
                        center_distance = np.sqrt(
                            (pred_center_x - prev_center_x) ** 2 +
                            (pred_center_y - prev_center_y) ** 2
                        )
                        # Normalize by image diagonal (max possible distance)
                        image_diagonal = np.sqrt(image_width ** 2 + image_height ** 2)
                        normalized_distance = center_distance / image_diagonal if image_diagonal > 0 else 1.0
                        # Convert distance to similarity (closer = higher score)
                        # Use exponential decay: exp(-distance * scale_factor)
                        distance_similarity = np.exp(-normalized_distance * 5.0)  # Scale factor 5.0
                    else:
                        center_distance = float('inf')
                        normalized_distance = 1.0
                        distance_similarity = 0.0

                    # 3. SAM confidence score (already normalized 0-1)
                    sam_score = float(scores[mask_idx])

                    # Combined score with weights:
                    # - 50% Center distance similarity (proximity) - most important for tracking
                    # - 30% IoU (overlap with previous mask)
                    # - 20% SAM confidence (quality)
                    combined_score = 0.5 * distance_similarity + 0.3 * iou + 0.2 * sam_score

                    mask_candidates.append({
                        'idx': mask_idx,
                        'iou': iou,
                        'center_distance': center_distance,
                        'normalized_distance': normalized_distance,
                        'distance_similarity': distance_similarity,
                        'sam_score': sam_score,
                        'combined_score': combined_score,
                    })

                # Sort by combined score (descending)
                mask_candidates.sort(key=lambda x: x['combined_score'], reverse=True)

                # Log top 3 candidates for debugging
                log_info(f"[SAM_TRACKER] State {idx} Top 3 mask candidates:")
                for rank, candidate in enumerate(mask_candidates[:3], 1):
                    log_info(
                        f"[SAM_TRACKER] State {idx} Rank {rank}: mask {candidate['idx']} - "
                        f"IoU={candidate['iou']:.4f}, "
                        f"center_dist={candidate['center_distance']:.1f}px ({candidate['normalized_distance']:.4f} normalized), "
                        f"dist_sim={candidate['distance_similarity']:.4f}, "
                        f"SAM_score={candidate['sam_score']:.4f}, "
                        f"combined={candidate['combined_score']:.4f}"
                    )

                # Select the best mask
                best_idx = mask_candidates[0]['idx']
                best_score = mask_candidates[0]['combined_score']
                log_info(f"[SAM_TRACKER] State {idx} Selected mask {best_idx} with combined score {best_score:.4f}")

                # Get best mask (use the selected index)
                best_mask = masks[best_idx] > 0.0
                log_info(f"[SAM_TRACKER] State {idx} best mask: shape={best_mask.shape}, sum={best_mask.sum()}")

                # Determine shape type from original (stored in state or default to mask)
                shape_type = state.get("shape_type", "mask")
                new_shape = self._mask_to_shape(best_mask, shape_type, image_width, image_height)

                if new_shape:
                    new_shape["label"] = state.get("label", "object")
                    log_info(f"[SAM_TRACKER] State {idx} converted to output: type={new_shape.get('type')}, points_count={len(new_shape.get('points', []))}")
                    new_shapes.append(new_shape)

                    # Update state
                    mask_list = best_mask.astype(np.uint8).tolist()
                    new_state = {
                        "mask": mask_list,
                        "frame": frame,
                        "shape_type": shape_type,
                        "label": new_shape["label"],
                        "image_height": image_height,
                        "image_width": image_width,
                    }
                    new_states.append(new_state)
                    log_info(f"[SAM_TRACKER] State {idx} updated: frame={frame}, mask_shape={len(mask_list)}x{len(mask_list[0]) if mask_list else 0}")
                else:
                    log_warning(f"[SAM_TRACKER] State {idx} failed to convert mask to shape")
                    new_shapes.append(None)
                    new_states.append(None)

        log_info(f"[SAM_TRACKER] handle completed: returning {len(new_shapes)} shape(s), {len(new_states)} state(s)")
        return new_shapes, new_states

