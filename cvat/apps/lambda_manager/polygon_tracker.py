# Copyright (C) CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT

"""
Polygon Track Builder for Mask R-CNN Detection
================================================

This module implements server-side track building for polygon/mask annotations.
It associates detections across video frames using:
1. IoU (Intersection over Union) matching between masks/polygons
2. Spatial proximity matching
3. Temporal continuity constraints

Usage:
    This is called from LambdaJob when polygon tracking mode is enabled.
"""

import base64
import json
import logging
import time
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment

from cvat.apps.dataset_manager import task as dm_task
from cvat.apps.dataset_manager.task import PatchAction
from cvat.apps.engine.frame_provider import TaskFrameProvider
from cvat.apps.engine.log import ServerLogManager
from cvat.apps.engine.models import Job, SourceType, Task
from cvat.apps.engine.serializers import LabeledDataSerializer
from cvat.apps.lambda_manager.views import (
    DetectionResultConverter,
    LambdaFunction,
)
from cvat.apps.lambda_manager.rq import LambdaRQMeta

slogger = ServerLogManager(__name__)


class PolygonTrackBuilder:
    """
    Builds polygon tracks from frame-by-frame mask detections.

    Algorithm:
    1. Run detection on all frames
    2. Convert masks to polygons if needed
    3. Associate detections across frames using:
       - IoU matching for masks
       - Spatial proximity (Hungarian matching)
       - Temporal continuity
    4. Create PolygonTrack objects
    5. Submit to CVAT
    """

    def __init__(self, db_task: Task, db_job: Optional[Job] = None):
        self.db_task = db_task
        self.db_job = db_job
        self.converter = DetectionResultConverter(db_task)
        self.frame_provider = TaskFrameProvider(db_task)
        self._last_progress_update = 0.0
        self._progress_update_interval = 5.0

    def _get_frame_set(self) -> List[int]:
        """Get the list of frames to process."""
        if self.db_job:
            task_data = self.db_task.data
            data_start_frame = task_data.start_frame
            step = task_data.get_frame_step()
            frame_set = sorted(
                (abs_id - data_start_frame) // step for abs_id in self.db_job.segment.frame_set
            )
        else:
            frame_set = range(self.db_task.data.size)

        return list(frame_set)

    def _run_detections(
        self,
        function: LambdaFunction,
        frame_set: List[int],
        threshold: float,
        mapping: Optional[dict],
        conv_mask_to_poly: bool,
    ) -> Dict[int, dict]:
        """
        Run detection on all frames and store results.

        Returns: {frame: {"shapes": [...]}}
        """
        slogger.glob.info(f"Running detection on {len(frame_set)} frames")

        frame_detections = {}
        total_frames = len(frame_set)

        for idx, frame in enumerate(frame_set):
            if frame in self.db_task.data.deleted_frames:
                continue

            try:
                # Use function.invoke() which properly handles detector functions
                # This ensures requests go to the correct detector container, not tracker
                # NOTE: For batch polygon tracking, we use the DETECTOR function (not tracker)
                # The tracker function is for interactive tracking only
                slogger.glob.info(f"[POLYGON_TRACKER] Calling detector function: {function.id} (kind: {function.kind}) for frame {frame}")
                annotations = function.invoke(
                    self.db_task,
                    db_job=self.db_job,
                    data={
                        "frame": frame,
                        "mapping": mapping,
                        "threshold": threshold,
                        "conv_mask_to_poly": True,  # Always convert to polygon for tracking
                        "tracking_mode": "video" if self.db_task.data.get_frame_step() == 1 else "image",
                    },
                    converter=self.converter,
                )
                slogger.glob.info(f"[POLYGON_TRACKER] Detector function {function.id} returned {len(annotations.get('shapes', []))} shapes for frame {frame}")

                # function.invoke() returns converted annotations with "tags" and "shapes"
                frame_detections[frame] = annotations
                shapes = annotations.get('shapes', [])
                shapes_count = len(shapes)

                # Verify shapes are polygons (not masks)
                shape_types = [s.get('type', 'unknown') for s in shapes]
                mask_count = sum(1 for t in shape_types if t == 'mask')
                polygon_count = sum(1 for t in shape_types if t == 'polygon')

                if mask_count > 0:
                    slogger.glob.warning(f"Frame {frame}: Found {mask_count} mask shapes (expected polygons). Conversion may have failed.")
                slogger.glob.info(f"Frame {frame}: {shapes_count} detections ({polygon_count} polygons, {mask_count} masks)")

            except Exception as e:
                slogger.glob.error(f"Error detecting frame {frame}: {e}")
                frame_detections[frame] = {"shapes": [], "tracks": []}

            # Update progress
            progress = (idx + 1) / total_frames
            self._update_progress(progress * 0.4)  # Detection is 40% of total work

        return frame_detections

    def _calculate_iou_mask(self, mask1_points: List[float], mask2_points: List[float],
                            image_width: int, image_height: int) -> float:
        """
        Calculate IoU between two masks given their flattened pixel format.

        Mask format: [pixel1, pixel2, ..., x_min, y_min, x_max, y_max]
        """
        try:
            # Extract bounding boxes
            if len(mask1_points) < 6 or len(mask2_points) < 6:
                return 0.0

            x1_min, y1_min, x1_max, y1_max = [int(x) for x in mask1_points[-4:]]
            x2_min, y2_min, x2_max, y2_max = [int(x) for x in mask2_points[-4:]]

            # Extract pixel data
            pixels1 = mask1_points[:-4]
            pixels2 = mask2_points[:-4]

            # Calculate dimensions
            w1 = x1_max - x1_min + 1
            h1 = y1_max - y1_min + 1
            w2 = x2_max - x2_min + 1
            h2 = y2_max - y2_min + 1

            if len(pixels1) != w1 * h1 or len(pixels2) != w2 * h2:
                return 0.0

            # Reshape to 2D masks
            mask1 = np.array(pixels1, dtype=np.uint8).reshape((h1, w1))
            mask2 = np.array(pixels2, dtype=np.uint8).reshape((h2, w2))

            # Create full-size masks
            full_mask1 = np.zeros((image_height, image_width), dtype=np.uint8)
            full_mask2 = np.zeros((image_height, image_width), dtype=np.uint8)

            full_mask1[y1_min:y1_max+1, x1_min:x1_max+1] = mask1
            full_mask2[y2_min:y2_max+1, x2_min:x2_max+1] = mask2

            # Calculate IoU
            intersection = np.logical_and(full_mask1, full_mask2).sum()
            union = np.logical_or(full_mask1, full_mask2).sum()

            return float(intersection / union) if union > 0 else 0.0

        except Exception as e:
            slogger.glob.warning(f"Error calculating IoU: {e}")
            return 0.0

    def _calculate_polygon_iou(self, poly1_points: List[float], poly2_points: List[float],
                               image_width: int, image_height: int) -> float:
        """
        Calculate IoU between two polygons by converting to masks.

        Polygon format: [x1, y1, x2, y2, x3, y3, ...]
        """
        try:
            if len(poly1_points) < 6 or len(poly2_points) < 6:
                return 0.0

            # Convert polygons to masks
            mask1 = np.zeros((image_height, image_width), dtype=np.uint8)
            mask2 = np.zeros((image_height, image_width), dtype=np.uint8)

            # Reshape points to (N, 2)
            points1 = np.array(poly1_points).reshape(-1, 2).astype(np.int32)
            points2 = np.array(poly2_points).reshape(-1, 2).astype(np.int32)

            cv2.fillPoly(mask1, [points1], 255)
            cv2.fillPoly(mask2, [points2], 255)

            # Calculate IoU
            intersection = np.logical_and(mask1, mask2).sum()
            union = np.logical_or(mask1, mask2).sum()

            return float(intersection / union) if union > 0 else 0.0

        except Exception as e:
            slogger.glob.warning(f"Error calculating polygon IoU: {e}")
            return 0.0

    def _calculate_shape_center(self, shape: dict) -> Optional[Tuple[float, float]]:
        """Calculate the center of a shape (polygon or mask)."""
        points = shape.get("points", [])
        if not points:
            return None

        shape_type = shape.get("type", "")

        if shape_type == "polygon":
            # Polygon: [x1, y1, x2, y2, ...]
            if len(points) < 6:
                return None
            points_array = np.array(points).reshape(-1, 2)
            center = np.mean(points_array, axis=0)
            return tuple(center)

        elif shape_type == "mask":
            # Mask: [pixel1, pixel2, ..., x_min, y_min, x_max, y_max]
            if len(points) < 6:
                return None
            x_min, y_min, x_max, y_max = [float(x) for x in points[-4:]]
            center = ((x_min + x_max) / 2, (y_min + y_max) / 2)
            return center

        return None

    def _calculate_iou(self, shape1: dict, shape2: dict, image_width: int, image_height: int) -> float:
        """Calculate IoU between two shapes."""
        points1 = shape1.get("points", [])
        points2 = shape2.get("points", [])
        type1 = shape1.get("type", "")
        type2 = shape2.get("type", "")

        # Both must be same type for meaningful comparison
        if type1 != type2:
            return 0.0

        if type1 == "polygon":
            return self._calculate_polygon_iou(points1, points2, image_width, image_height)
        elif type1 == "mask":
            return self._calculate_iou_mask(points1, points2, image_width, image_height)

        return 0.0

    def _associate_detections(
        self,
        frame_detections: Dict[int, dict],
        max_distance: float = 150.0,
        iou_threshold: float = 0.3,
        max_frame_gap: int = 5,
    ) -> Dict[int, List[Dict]]:
        """
        Associate detections across frames to build tracks.

        Algorithm:
        - For each frame, match detections with previous frame's tracks
        - Use Hungarian algorithm for optimal matching based on IoU
        - Track ID is maintained across frames
        - Handles gaps (missing detections) up to max_frame_gap frames

        Returns: {track_id: [{"frame": frame, "shape": shape}, ...]}
        """
        slogger.glob.info("Associating detections across frames")

        # Get image dimensions (assume same for all frames)
        frame_set = sorted(frame_detections.keys())
        if not frame_set:
            return {}

        # Get image dimensions from first frame
        # Use PIL output type to get image dimensions
        from cvat.apps.engine.frame_provider import FrameOutputType
        image_data = self.frame_provider.get_frame(
            frame_set[0],
            out_type=FrameOutputType.PIL
        )
        # image_data.data is a PIL Image
        image_width, image_height = image_data.data.size

        # Group detections by label
        tracks_by_label = {}  # {label_id: {track_id: [(frame, shape)]}}
        next_track_id = 0

        for frame_idx, frame in enumerate(frame_set):
            frame_data = frame_detections[frame]
            shapes = frame_data.get("shapes", [])

            if not shapes:
                continue

            # Group shapes by label for matching
            shapes_by_label = {}
            for shape in shapes:
                label_id = shape["label_id"]
                if label_id not in shapes_by_label:
                    shapes_by_label[label_id] = []
                shapes_by_label[label_id].append(shape)

            # Process each label separately
            for label_id, current_shapes in shapes_by_label.items():
                # Initialize label tracks if needed
                if label_id not in tracks_by_label:
                    tracks_by_label[label_id] = {}

                label_tracks = tracks_by_label[label_id]

                # Get active tracks (tracks that can be extended)
                # Include tracks from previous frames (frame_gap >= 1) up to max_frame_gap
                # This includes both consecutive frames (gap=1) and tracks with small gaps
                active_tracks = []
                for track_id, track_shapes in label_tracks.items():
                    if not track_shapes:
                        continue
                    last_frame, _ = track_shapes[-1]
                    frame_gap = frame - last_frame
                    # Match with tracks from previous frames (gap >= 1) up to max_frame_gap
                    # gap=1 means consecutive frames, gap>1 means there was a gap
                    if 1 <= frame_gap <= max_frame_gap:
                        active_tracks.append((track_id, track_shapes))

                if not active_tracks and not current_shapes:
                    continue

                # If no active tracks, create new tracks for all current shapes
                if not active_tracks:
                    for shape in current_shapes:
                        track_id = next_track_id
                        next_track_id += 1
                        label_tracks[track_id] = [(frame, shape)]
                    continue

                # Build cost matrix for Hungarian algorithm
                # Rows: active tracks, Columns: current detections
                num_tracks = len(active_tracks)
                num_detections = len(current_shapes)

                if num_tracks == 0 or num_detections == 0:
                    # No matches possible, create new tracks for detections
                    for shape in current_shapes:
                        track_id = next_track_id
                        next_track_id += 1
                        label_tracks[track_id] = [(frame, shape)]
                    continue

                # Cost matrix: 1 - IoU (lower is better)
                cost_matrix = np.ones((num_tracks, num_detections)) * 1000.0  # Large penalty

                for i, (track_id, track_shapes) in enumerate(active_tracks):
                    last_frame, last_shape = track_shapes[-1]
                    frame_gap = frame - last_frame

                    for j, current_shape in enumerate(current_shapes):
                        # Calculate IoU
                        iou = self._calculate_iou(last_shape, current_shape, image_width, image_height)

                        # Adjust cost based on frame gap (prefer closer frames)
                        gap_penalty = 1.0 + (frame_gap - 1) * 0.1
                        cost = (1.0 - iou) * gap_penalty

                        # Only consider if IoU is above threshold
                        if iou >= iou_threshold:
                            cost_matrix[i, j] = cost
                        else:
                            # Log low IoU for debugging
                            if frame_gap == 1:  # Consecutive frames should have higher IoU
                                slogger.glob.debug(
                                    f"Low IoU ({iou:.3f} < {iou_threshold}) for consecutive frames "
                                    f"(track {track_id}, frame {last_frame} -> {frame})"
                                )

                # Solve assignment problem
                track_indices, detection_indices = linear_sum_assignment(cost_matrix)

                # Process matches
                matched_detections = set()
                for track_idx, det_idx in zip(track_indices, detection_indices):
                    cost = cost_matrix[track_idx, det_idx]
                    if cost < 1000.0:  # Valid match
                        track_id, _ = active_tracks[track_idx]
                        matched_shape = current_shapes[det_idx]
                        label_tracks[track_id].append((frame, matched_shape))
                        matched_detections.add(det_idx)

                # Create new tracks for unmatched detections
                for det_idx, shape in enumerate(current_shapes):
                    if det_idx not in matched_detections:
                        track_id = next_track_id
                        next_track_id += 1
                        label_tracks[track_id] = [(frame, shape)]

        # Flatten structure for return
        all_tracks = []
        total_detections = 0
        for label_id, label_tracks in tracks_by_label.items():
            for track_id, frame_shapes in label_tracks.items():
                if frame_shapes:
                    all_tracks.append({
                        "label_id": label_id,
                        "track_id": track_id,
                        "frames": frame_shapes,
                    })
                    total_detections += len(frame_shapes)

        # Log statistics
        avg_detections_per_track = total_detections / len(all_tracks) if all_tracks else 0
        slogger.glob.info(
            f"Created {len(all_tracks)} tracks from {len(frame_set)} frames "
            f"({total_detections} total detections, avg {avg_detections_per_track:.1f} per track)"
        )

        # Warn if too many tracks (suggests poor association)
        if len(frame_set) > 0:
            first_frame_shapes = frame_detections.get(frame_set[0], {}).get("shapes", [])
            expected_tracks = len(first_frame_shapes)
            if expected_tracks > 0 and len(all_tracks) > expected_tracks * 3:
                slogger.glob.warning(
                    f"Created {len(all_tracks)} tracks but expected ~{expected_tracks} "
                    f"(one per unique object). This suggests association may not be working correctly. "
                    f"Consider adjusting iou_threshold (current: {iou_threshold}) or max_frame_gap (current: {max_frame_gap})"
                )

        return all_tracks

    def _convert_tracks_to_cvat_format(self, raw_tracks: List[dict]) -> List[dict]:
        """
        Convert internal track format to CVAT PolygonTrack format.
        """
        cvat_tracks = []

        for track_data in raw_tracks:
            label_id = track_data["label_id"]
            frame_shapes = track_data["frames"]

            # Sort by frame
            frame_shapes.sort(key=lambda x: x[0])

            if not frame_shapes:
                continue

            # Build track
            track = {
                "label_id": label_id,
                "frame": frame_shapes[0][0],
                "shapes": [],
                "source": str(SourceType.AUTO),
                "group": None,
                "attributes": [],
            }

            # Add shapes for each frame
            for frame, shape in frame_shapes:
                # Ensure shape is polygon type (should already be converted)
                shape_type = shape.get("type", "polygon")

                # If it's still a mask, try to convert it
                if shape_type == "mask":
                    slogger.glob.warning(f"Found mask in track at frame {frame}, attempting conversion")
                    from cvat.apps.lambda_manager.views import DetectionResultConverter
                    polygon_points = DetectionResultConverter._mask_to_polygon(shape.get("points", []))
                    if polygon_points is not None:
                        shape["type"] = "polygon"
                        shape["points"] = polygon_points
                        slogger.glob.info(f"Successfully converted mask to polygon at frame {frame}")
                    else:
                        slogger.glob.warning(f"Failed to convert mask to polygon at frame {frame}, skipping")
                        continue

                # Ensure we have polygon points
                if shape_type != "polygon" or not shape.get("points"):
                    slogger.glob.warning(f"Invalid shape type or missing points at frame {frame}: type={shape_type}")
                    continue

                track["shapes"].append({
                    "frame": frame,
                    "type": "polygon",  # Explicitly set to polygon
                    "occluded": shape.get("occluded", False),
                    "outside": shape.get("outside", False),
                    "points": shape.get("points", []),
                    "z_order": shape.get("z_order", 0),
                    "rotation": shape.get("rotation", 0),
                    "attributes": shape.get("attributes", []),
                })

            # Add final shape with outside=True if track doesn't end at last frame
            if frame_shapes:
                last_frame, last_shape = frame_shapes[-1]
                # Check if we need to add an outside shape
                # This would require knowing the last frame in the task, which we can get
                frame_set = self._get_frame_set()
                if frame_set and last_frame < frame_set[-1]:
                    track["shapes"].append({
                        "frame": last_frame + 1,
                        "type": "polygon",
                        "occluded": False,
                        "outside": True,
                        "points": last_shape.get("points", []),
                        "z_order": last_shape.get("z_order", 0),
                        "rotation": last_shape.get("rotation", 0),
                        "attributes": last_shape.get("attributes", []),
                    })

            cvat_tracks.append(track)

        return cvat_tracks

    def _get_image(self, frame: int) -> str:
        """Get base64-encoded image for a frame."""
        image = self.frame_provider.get_frame(frame)
        return base64.b64encode(image.data.getvalue()).decode("utf-8")

    def _update_progress(self, progress: float):
        """Update RQ job progress with throttling."""
        current_time = time.time()
        # Update if enough time has passed or if progress is 0 or 1.0 (start/end)
        if (current_time - self._last_progress_update) >= self._progress_update_interval or progress in (0.0, 1.0):
            from rq import get_current_job
            job = get_current_job()
            if job:
                rq_job_meta = LambdaRQMeta.for_job(job)
                rq_job_meta.progress = int(progress * 100)
                rq_job_meta.save()
            self._last_progress_update = current_time

    def build_and_submit_tracks(
        self,
        function: LambdaFunction,
        threshold: float,
        mapping: Optional[dict],
        conv_mask_to_poly: bool,
        max_distance: float = 150.0,
        iou_threshold: float = 0.3,
        max_frame_gap: int = 5,
    ) -> None:
        """Main method: build polygon tracks and submit to CVAT with error handling."""
        slogger.glob.info(f"Starting polygon track building for task {self.db_task.id}")

        frame_detections = None
        raw_tracks = None
        cvat_tracks = None

        try:
            # Step 1: Get frame set
            frame_set = self._get_frame_set()
            if not frame_set:
                slogger.glob.info("No frames to process")
                return

            # Step 2: Run detections on all frames
            self._update_progress(0.0)
            frame_detections = self._run_detections(
                function, frame_set, threshold, mapping, conv_mask_to_poly
            )

            if not frame_detections or not any(frame_detections.values()):
                slogger.glob.warning("No detections found in any frame")
                return

            # Step 3: Associate detections into tracks
            self._update_progress(0.5)
            raw_tracks = self._associate_detections(
                frame_detections, max_distance, iou_threshold, max_frame_gap
            )

            if not raw_tracks:
                slogger.glob.info("No tracks created from detections")
                return

            # Step 4: Convert to CVAT format
            self._update_progress(0.85)
            cvat_tracks = self._convert_tracks_to_cvat_format(raw_tracks)

            if not cvat_tracks:
                slogger.glob.warning("No tracks converted to CVAT format")
                return

            # Step 5: Submit to CVAT
            self._update_progress(0.95)

            # Verify all shapes in tracks are polygons
            for track in cvat_tracks:
                for shape in track.get("shapes", []):
                    if shape.get("type") != "polygon":
                        slogger.glob.error(f"Track shape has wrong type: {shape.get('type')}, expected polygon. Frame: {shape.get('frame')}")
                        shape["type"] = "polygon"  # Force to polygon

            data = {
                "tracks": cvat_tracks,
                "shapes": [],
                "tags": [],
            }

            slogger.glob.info(f"Submitting {len(cvat_tracks)} tracks with polygon shapes to CVAT")
            for i, track in enumerate(cvat_tracks[:3]):  # Log first 3 tracks
                shape_types = [s.get("type") for s in track.get("shapes", [])]
                slogger.glob.info(f"Track {i}: label_id={track.get('label_id')}, {len(track.get('shapes', []))} shapes, types={set(shape_types)}")

            serializer = LabeledDataSerializer(data=data)
            if serializer.is_valid(raise_exception=True):
                if self.db_job:
                    dm_task.patch_job_data(self.db_job.id, serializer.data, PatchAction.CREATE)
                else:
                    dm_task.patch_task_data(self.db_task.id, serializer.data, PatchAction.CREATE)
                slogger.glob.info(f"Successfully submitted {len(cvat_tracks)} polygon tracks to CVAT")
            else:
                slogger.glob.error(f"Serializer validation failed: {serializer.errors}")

            self._update_progress(1.0)
            slogger.glob.info(f"Successfully submitted {len(cvat_tracks)} polygon tracks to CVAT")

        except Exception as e:
            # Log error with full context
            slogger.glob.error(
                f"Polygon tracking failed for task {self.db_task.id}: {e}",
                exc_info=True
            )

            # Try to save partial results if we have tracks
            if cvat_tracks and len(cvat_tracks) > 0:
                try:
                    slogger.glob.info(f"Attempting to save {len(cvat_tracks)} partial tracks")
                    data = {"tracks": cvat_tracks, "shapes": [], "tags": []}
                    serializer = LabeledDataSerializer(data=data)
                    if serializer.is_valid(raise_exception=True):
                        if self.db_job:
                            dm_task.put_job_data(self.db_job.id, serializer.data)
                        else:
                            dm_task.put_task_data(self.db_task.id, serializer.data)
                    slogger.glob.info("Partial tracks saved successfully")
                except Exception as save_error:
                    slogger.glob.error(f"Failed to save partial tracks: {save_error}")

            # Re-raise to mark RQ job as failed
            raise

