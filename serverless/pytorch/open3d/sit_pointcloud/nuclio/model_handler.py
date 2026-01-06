import io
import logging
import math
import os
import tempfile
from typing import List, Optional, Tuple

import numpy as np
import open3d as o3d
from sklearn.decomposition import PCA


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

            confidence = self._compute_confidence(cluster_pts, extent)

            if confidence < threshold:
                continue

            label = self._label_for_extent(extent)
            if not label:  # Skip if label is empty (doesn't match task labels)
                continue

            center = cluster_pts.mean(axis=0)

            # Compute heading using PCA for direction estimation
            heading = self._compute_heading(cluster_pts)

            detections.append(
                {
                    "confidence": f"{confidence:.4f}",
                    "label": label,
                    "type": "cuboid",
                    "points": [
                        float(center[0]),
                        float(center[1]),
                        float(center[2]),
                        0.0,  # Rotation around X (roll) - keep 0
                        0.0,  # Rotation around Y (pitch) - keep 0
                        float(heading),  # Rotation around Z (yaw) - use computed heading
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

    def _compute_confidence(self, cluster_pts: np.ndarray, extent: np.ndarray) -> float:
        """Compute confidence based on point density and cluster quality."""
        num_points = cluster_pts.shape[0]

        # Point density (points per unit volume)
        volume = extent[0] * extent[1] * extent[2]
        density = num_points / volume if volume > 0 else 0

        # Normalize confidence (adjust thresholds as needed)
        min_density = 10.0  # Minimum expected points per cubic meter
        max_density = 1000.0  # Maximum expected points per cubic meter

        normalized_density = (density - min_density) / (max_density - min_density)
        normalized_density = np.clip(normalized_density, 0, 1)

        return float(normalized_density)

    def _compute_heading(self, cluster_pts: np.ndarray) -> float:
        """Compute heading angle using PCA on XY plane.

        Returns:
            float: Heading angle in radians (yaw around Z-axis)
                  Positive values indicate counter-clockwise rotation from +X axis
        """
        try:
            # Use only X and Y coordinates for heading
            xy_pts = cluster_pts[:, :2]

            # Perform PCA to find principal direction
            pca = PCA(n_components=2)
            pca.fit(xy_pts)

            # Get first principal component (direction of maximum variance)
            principal_component = pca.components_[0]

            # Compute heading angle in radians
            # arctan2(y, x) returns angle from +X axis
            heading = np.arctan2(principal_component[1], principal_component[0])

            return float(heading)
        except Exception as e:
            self.logger.warning(f"Failed to compute heading: {e}")
            return 0.0

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
        """Determine label based on object dimensions.

        Returns:
            str: "Pedestrian" if matches pedestrian dimensions,
                 "" (empty string) otherwise (to skip non-matching objects)
        """
        sorted_xy = sorted(extent[:2], reverse=True)
        length = sorted_xy[0]
        width = sorted_xy[1]
        height = extent[2]

        # Check if object matches pedestrian dimensions
        if (
            length <= self.pedestrian_max_length
            and width <= self.pedestrian_max_width
            and height <= self.pedestrian_max_height
        ):
            return "Pedestrian"  # Capital P to match task label

        # Return empty string for non-pedestrian objects
        # This prevents creating annotations for cars/trucks that aren't in your labels
        return ""

    def _load_points(self, raw: bytes) -> np.ndarray:
        """Load point cloud from binary data.

        Supports:
        - KITTI BIN format: float32 array [x, y, z, intensity]
        - PCD format: ASCII or binary
        """
        # Distinguish between PCD (ASCII/binary) and KITTI BIN (float32) sources.
        header = raw[:16]
        if header.startswith(b"VERSION") or header.startswith(b"# .PCD"):
            return self._read_pcd(raw)

        try:
            # Try KITTI BIN format first
            arr = np.frombuffer(raw, dtype=np.float32)
            if arr.size % 4 != 0:
                arr = arr[: arr.size - (arr.size % 4)]
            # KITTI format: [x, y, z, intensity], take only XYZ
            return arr.reshape(-1, 4)[:, :3]
        except ValueError:
            # Fall back to PCD path if reshape fails
            return self._read_pcd(raw)

    def _read_pcd(self, raw: bytes) -> np.ndarray:
        """Read PCD format point cloud."""
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
