import io
import logging
import math
import os
import tempfile
from typing import List, Optional, Tuple

import numpy as np
import open3d as o3d


class PointCloudDetector:
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.eps = float(os.getenv("DBSCAN_EPS", "1.2"))
        self.min_points = int(os.getenv("DBSCAN_MIN_CLUSTER_POINTS", "40"))
        self.max_clusters = int(os.getenv("MAX_CLUSTER_COUNT", "64"))
        self.voxel_size = float(os.getenv("VOXEL_SIZE", "0.15"))
        self.default_threshold = float(os.getenv("DEFAULT_CONFIDENCE_THRESHOLD", "0.3"))
        self.confidence_norm = float(os.getenv("CONFIDENCE_NORMALIZER", "256"))
        self.pedestrian_max_length = float(os.getenv("PEDESTRIAN_MAX_LENGTH", "1.5"))
        self.pedestrian_max_width = float(os.getenv("PEDESTRIAN_MAX_WIDTH", "1.0"))
        self.pedestrian_max_height = float(os.getenv("PEDESTRIAN_MAX_HEIGHT", "2.0"))
        self.vehicle_max_length = float(os.getenv("VEHICLE_MAX_LENGTH", "5.0"))
        self.vehicle_max_height = float(os.getenv("VEHICLE_MAX_HEIGHT", "3.0"))

    def infer(self, *, cloud_bytes: bytes, threshold: Optional[float], frame_id: int) -> List[dict]:
        threshold = float(threshold) if threshold is not None else self.default_threshold

        points = self._load_points(cloud_bytes)
        if points.size == 0:
            self.logger.debug("Frame %s contains no points after decoding", frame_id)
            return []

        clustered_points, clusters = self._cluster(points)
        detections: List[dict] = []
        for cluster_id in np.unique(clusters):
            if cluster_id < 0:
                continue

            mask = clusters == cluster_id
            cluster_pts = clustered_points[mask]
            if cluster_pts.shape[0] < self.min_points:
                continue

            extent = cluster_pts.max(axis=0) - cluster_pts.min(axis=0)
            if not np.all(np.isfinite(extent)) or np.any(extent <= 0):
                continue

            confidence = min(
                0.99, cluster_pts.shape[0] / (self.confidence_norm + cluster_pts.shape[0])
            )

            if confidence < threshold:
                continue

            label = self._label_for_extent(extent)
            center = cluster_pts.mean(axis=0)

            detections.append(
                {
                    "confidence": f"{confidence:.4f}",
                    "label": label,
                    "type": "cuboid",
                    "points": [
                        float(center[0]),
                        float(center[1]),
                        float(center[2]),
                        0.0,
                        0.0,
                        0.0,
                        float(extent[0]),
                        float(extent[1]),
                        float(extent[2]),
                    ],
                }
            )

            if len(detections) >= self.max_clusters:
                break

        self.logger.debug(
            "Frame %s -> %d cuboids (threshold %.2f)", frame_id, len(detections), threshold
        )
        return detections

    def _cluster(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        if self.voxel_size > 0:
            pcd = pcd.voxel_down_sample(self.voxel_size)
        down_pts = np.asarray(pcd.points)
        if down_pts.size == 0:
            return down_pts, np.empty(0, dtype=int)
        labels = np.array(
            pcd.cluster_dbscan(
                eps=self.eps, min_points=self.min_points, print_progress=False
            )
        )
        return down_pts, labels

    def _label_for_extent(self, extent: np.ndarray) -> str:
        sorted_xy = sorted(extent[:2], reverse=True)
        length = sorted_xy[0]
        width = sorted_xy[1]
        height = extent[2]
        if (
            length <= self.pedestrian_max_length
            and width <= self.pedestrian_max_width
            and height <= self.pedestrian_max_height
        ):
            return "pedestrian"
        if length <= self.vehicle_max_length and height <= self.vehicle_max_height:
            return "car"
        return "truck"

    def _load_points(self, raw: bytes) -> np.ndarray:
        # Distinguish between PCD (ASCII/binary) and KITTI BIN (float32) sources.
        header = raw[:16]
        if header.startswith(b"VERSION") or header.startswith(b"# .PCD"):
            return self._read_pcd(raw)

        try:
            arr = np.frombuffer(raw, dtype=np.float32)
            if arr.size % 4 != 0:
                arr = arr[: arr.size - (arr.size % 4)]
            return arr.reshape(-1, 4)[:, :3]
        except ValueError:
            # Fall back to PCD path if reshape fails
            return self._read_pcd(raw)

    def _read_pcd(self, raw: bytes) -> np.ndarray:
        with tempfile.NamedTemporaryFile(suffix=".pcd", delete=False) as tmp:
            tmp.write(raw)
            tmp.flush()
            tmp_path = tmp.name

        try:
            pcd = o3d.io.read_point_cloud(tmp_path)
            pts = np.asarray(pcd.points)
        except Exception as exc:
            self.logger.warning("Failed to parse PCD payload: %s", exc)
            pts = np.empty((0, 3), dtype=np.float32)
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

        return pts

