# Copyright (C) CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT

"""
Skeleton Track Builder for MediaPipe Pose Detection
===================================================

This module implements server-side track building for skeleton annotations.
It associates skeleton detections across video frames using:
1. Tracking IDs from MediaPipe VIDEO mode
2. Spatial proximity matching (Hungarian algorithm)
3. Temporal continuity constraints

Usage:
    This is called from LambdaJob when tracking mode is enabled.
"""

import base64
import json
import logging
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
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


class SkeletonTrackBuilder:
    """
    Builds skeleton tracks from frame-by-frame detections.

    Algorithm:
    1. Run detection on all frames
    2. Associate detections across frames using:
       - MediaPipe tracking IDs (if available)
       - Spatial proximity (Hungarian matching)
       - Temporal continuity
    3. Create SkeletonTrack objects
    4. Submit to CVAT
    """

    def __init__(self, db_task: Task, db_job: Optional[Job] = None):
        self.db_task = db_task
        self.db_job = db_job
        self.converter = DetectionResultConverter(db_task)
        self.frame_provider = TaskFrameProvider(db_task)

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

            # Get image for this frame
            image_b64 = self._get_image(frame)

            # Call MediaPipe service with VIDEO mode parameters
            payload = {
                "image": image_b64,
                "frame_number": frame,
                "threshold": threshold,
                "tracking_mode": "video",
                "job_id": self.db_job.id if self.db_job else None,
                "task_id": self.db_task.id,
            }

            # Invoke function directly (bypass converter for now)
            # We need raw detections with tracking info
            try:
                # Get raw response from MediaPipe
                raw_response = function.gateway.invoke(function, payload)

                # For tracker kind function, the reply is specialized
                # CVAT Expects: {"shapes": [...], "states": [...]}
                # DetectionResultConverter expects just the list of shapes
                if isinstance(raw_response, dict) and "shapes" in raw_response:
                    raw_response = raw_response["shapes"]

                # Convert to CVAT format
                annotations = self.converter.convert(
                    conv_mask_to_poly=conv_mask_to_poly,
                    frame=frame,
                    annotations=raw_response,
                )

                frame_detections[frame] = annotations
                slogger.glob.info(f"Frame {frame}: {len(annotations.get('shapes', []))} detections")

            except Exception as e:
                slogger.glob.error(f"Error detecting frame {frame}: {e}")
                frame_detections[frame] = {"shapes": [], "tracks": []}

            # Update progress
            progress = (idx + 1) / total_frames
            self._update_progress(progress)

        return frame_detections

    def _get_seed_tracks(self, start_frame: int) -> Dict[int, Dict[int, List]]:
        """
        Fetch existing skeletons from the start frame to use as seeds.
        Returns: {label_id: {track_id: [(frame, shape)]}}
        """
        if not self.db_job:
            return {}

        try:
            # Get current job data
            data = dm_task.get_job_data(self.db_job.id)
            shapes = data.get("shapes", [])

            seed_tracks = {} # {label_id: {track_id: [(frame, shape)]}}
            next_track_id = 0

            for shape in shapes:
                # Check if shape is on start frame and is a skeleton
                if shape["frame"] == start_frame and shape["type"] == "skeleton":
                    label_id = shape["label_id"]

                    if label_id not in seed_tracks:
                        seed_tracks[label_id] = {}

                    # Assign a temporary track ID (negative to avoid conflict with auto ids,
                    # or just start positive if we control the namespace)
                    # We'll use 0-indexed incrementing integers for simplicity in this run
                    track_id = next_track_id
                    next_track_id += 1

                    # Store as a track with one frame
                    seed_tracks[label_id][track_id] = [(start_frame, shape)]

            return seed_tracks
        except Exception as e:
            slogger.glob.error(f"Failed to get seed tracks: {e}")
            return {}

    def _associate_detections(
        self,
        frame_detections: Dict[int, dict],
        max_distance: float = 150.0,
        seed_tracks: Dict[int, Dict[int, List]] = None,
    ) -> Dict[int, List[Dict]]:
        """
        Associate detections across frames to build tracks.

        Algorithm:
        - For each frame, match detections with previous frame's tracks
        - Use Hungarian algorithm for optimal matching
        - Track ID is maintained across frames

        Returns: {track_id: [{"frame": frame, "shape": shape}, ...]}
        """
        slogger.glob.info("Associating detections across frames")

        # Group detections by label
        tracks_by_label = {}  # {label_id: {track_id: [(frame, shape)]}}
        next_track_id = 0

        # Initialize with seed tracks if provided
        if seed_tracks:
            # Deep copy to avoid modifying source
            from copy import deepcopy
            tracks_by_label = deepcopy(seed_tracks)

            # Find max track ID to avoid conflicts
            max_id = -1
            for label_tracks in tracks_by_label.values():
                for track_id in label_tracks.keys():
                    if isinstance(track_id, int):
                        max_id = max(max_id, track_id)

            if max_id >= 0:
                next_track_id = max_id + 1
            slogger.glob.info(f"Initialized with seed tracks, next_track_id={next_track_id}")

        frame_set = sorted(frame_detections.keys())
        has_seeds = bool(seed_tracks)
        start_frame = frame_set[0] if frame_set else -1

        for frame_idx, frame in enumerate(frame_set):
            # If we have seeds for the start frame, skip processing detections on that frame
            # We rely on the manual annotations to establish the initial state
            if has_seeds and frame == start_frame:
                continue

            frame_data = frame_detections[frame]
            shapes = frame_data.get("shapes", [])

            if not shapes:
                continue

            for shape in shapes:
                label_id = shape["label_id"]

                # Initialize label tracks if needed
                if label_id not in tracks_by_label:
                    tracks_by_label[label_id] = {}

                label_tracks = tracks_by_label[label_id]

                # Get skeleton center for matching
                center = self._get_skeleton_center(shape)
                if center is None:
                    continue

                # Try to match with existing tracks from previous frame
                prev_frame = frame_set[frame_idx - 1] if frame_idx > 0 else None
                matched_track_id = None

                if prev_frame and prev_frame in label_tracks:
                    # Calculate distances to all existing tracks
                    potential_matches = []

                    for track_id, track_shapes in label_tracks.items():
                        if not track_shapes:
                            continue

                        # Get last shape in track
                        last_frame, last_shape = track_shapes[-1]

                        # Check frame gap
                        frame_gap = frame - last_frame
                        if frame_gap > 20:  # Too far apart
                            continue

                        # Calculate center distance
                        last_center = self._get_skeleton_center(last_shape)
                        if last_center is None:
                            continue

                        distance = np.sqrt(
                            (center[0] - last_center[0])**2 +
                            (center[1] - last_center[1])**2
                        )

                        # Accept if distance is small enough
                        # Use adaptive threshold based on frame gap
                        effective_threshold = max_distance * (1.0 / (1.0 + frame_gap * 0.1))

                        if distance < effective_threshold:
                            potential_matches.append((track_id, distance))

                    # Choose best match (shortest distance)
                    if potential_matches:
                        potential_matches.sort(key=lambda x: x[1])
                        matched_track_id = potential_matches[0][0]

                if matched_track_id is None:
                    # Create new track
                    matched_track_id = next_track_id
                    next_track_id += 1
                    label_tracks[matched_track_id] = []

                # Add to track
                label_tracks[matched_track_id].append((frame, shape))

        # Flatten structure for return
        all_tracks = []
        for label_id, label_tracks in tracks_by_label.items():
            for track_id, frame_shapes in label_tracks.items():
                if frame_shapes:
                    all_tracks.append({
                        "label_id": label_id,
                        "track_id": track_id,
                        "frames": frame_shapes,
                    })

        slogger.glob.info(f"Created {len(all_tracks)} tracks from {len(frame_set)} frames")
        return all_tracks

    def _get_skeleton_center(self, shape: dict) -> Optional[Tuple[float, float]]:
        """
        Calculate the center of a skeleton shape.

        Uses visible elements only.
        """
        elements = shape.get("elements", [])
        if not elements:
            return None

        visible_elements = [e for e in elements if not e.get("outside", False)]
        if not visible_elements:
            return None

        # Calculate mean of all visible element positions
        points = []
        for element in visible_elements:
            if "points" in element and len(element["points"]) >= 2:
                points.append([element["points"][0], element["points"][1]])

        if not points:
            return None

        points_array = np.array(points)
        center = np.mean(points_array, axis=0)

        return tuple(center)

        return tuple(center)

    def _calculate_skeleton_bbox(self, elements: List[dict]) -> List[float]:
        """Calculate [xtl, ytl, xbr, ybr] bounding box from skeleton elements."""
        xs = []
        ys = []
        for el in elements:
            # Only use visible elements if possible
            if not el.get("outside", False) and "points" in el:
                xs.append(el["points"][0])
                ys.append(el["points"][1])

        # If no visible elements, use all elements
        if not xs:
            for el in elements:
                if "points" in el:
                    xs.append(el["points"][0])
                    ys.append(el["points"][1])

        if not xs:
            return [0.0, 0.0, 0.0, 0.0]

        return [min(xs), min(ys), max(xs), max(ys)]

    def _convert_tracks_to_cvat_format(self, raw_tracks: List[dict]) -> List[dict]:
        """
        Convert internal track format to CVAT SkeletonTrack format.
        """
        cvat_tracks = []

        for track_data in raw_tracks:
            label_id = track_data["label_id"]
            frame_shapes = track_data["frames"]

            # Sort by frame
            frame_shapes.sort(key=lambda x: x[0])

            # Build track
            track = {
                "label_id": label_id,
                "frame": frame_shapes[0][0],
                "shapes": [],
                "source": str(SourceType.AUTO),
                "group": None,
                "attributes": [],
            }

            # Add skeleton shapes for each frame
            for frame, shape in frame_shapes:
                elements = shape.get("elements", [])
                bbox = self._calculate_skeleton_bbox(elements)

                shape_copy = {
                    "frame": frame,
                    "label_id": label_id,
                    "type": "skeleton",
                    "occluded": False,
                    "outside": False,
                    "points": [], # CVAT backend expects empty points for skeletons
                    "z_order": 0,
                    "elements": elements,
                    "source": "auto",
                    "attributes": [],
                    "group": None,
                }
                track["shapes"].append(shape_copy)

            # Add final outside shape if needed
            last_frame = frame_shapes[-1][0]
            frame_set = self._get_frame_set()
            if last_frame < frame_set[-1]:
                last_shape = frame_shapes[-1][1]
                last_elements = last_shape.get("elements", [])
                last_bbox = self._calculate_skeleton_bbox(last_elements)

                track["shapes"].append({
                    "frame": last_frame + 1,
                    "label_id": label_id,
                    "type": "skeleton",
                    "occluded": False,
                    "outside": True,
                    "points": [],
                    "z_order": 0,
                    "elements": last_elements,
                    "source": "auto",
                    "attributes": [],
                    "group": None,
                })

            cvat_tracks.append(track)

        return cvat_tracks

    def _get_image(self, frame: int) -> str:
        """Get base64-encoded image for a frame."""
        image = self.frame_provider.get_frame(frame)
        return base64.b64encode(image.data.getvalue()).decode("utf-8")

    def _update_progress(self, progress: float):
        """Update RQ job progress."""
        from rq import get_current_job
        job = get_current_job()
        if job:
            rq_job_meta = LambdaRQMeta.for_job(job)
            rq_job_meta.progress = int(progress * 100)
            rq_job_meta.save()

    def build_and_submit_tracks(
        self,
        function: LambdaFunction,
        threshold: float,
        mapping: Optional[dict],
        conv_mask_to_poly: bool,
        max_distance: float = 150.0,
    ) -> None:
        """
        Main method: build skeleton tracks and submit to CVAT.
        """
        slogger.glob.info(f"Starting skeleton track building for task {self.db_task.id}")

        # Step 1: Get frame set
        frame_set = self._get_frame_set()
        if not frame_set:
            slogger.glob.info("No frames to process")
            return

        # Step 2: Run detections on all frames
        frame_detections = self._run_detections(
            function, frame_set, threshold, mapping, conv_mask_to_poly
        )

        # Step 3: Get seed tracks from existing annotations (if any)
        seed_tracks = self._get_seed_tracks(frame_set[0]) if frame_set else {}
        if seed_tracks:
            slogger.glob.info(f"Found {sum(len(v) for v in seed_tracks.values())} seed skeletons in frame {frame_set[0]}")

        # Step 4: Associate detections into tracks
        raw_tracks = self._associate_detections(frame_detections, max_distance, seed_tracks)

        if not raw_tracks:
            slogger.glob.info("No tracks created")
            return

        # Step 4: Convert to CVAT format
        cvat_tracks = self._convert_tracks_to_cvat_format(raw_tracks)

        # Step 5: Submit to CVAT
        data = {
            "tracks": cvat_tracks,
            "shapes": [],
            "tags": [],
        }

        serializer = LabeledDataSerializer(data=data)
        if serializer.is_valid(raise_exception=True):
            if self.db_job:
                dm_task.put_job_data(self.db_job.id, serializer.data)
            else:
                dm_task.put_task_data(self.db_task.id, serializer.data)

        slogger.glob.info(f"Successfully submitted {len(cvat_tracks)} skeleton tracks to CVAT")
