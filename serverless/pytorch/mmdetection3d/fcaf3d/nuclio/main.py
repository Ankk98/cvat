import base64
import json
import tempfile
import os
from typing import List, Dict, Any, Optional

# MMDetection3D imports will be available in Docker container
try:
    import mmcv
    import mmengine
    from mmengine import Config
    from mmdet3d.apis import inference_detector, init_model
    from mmdet3d.registry import VISUALIZERS
    from mmdet3d.utils import register_all_modules
    MMDetection3D_AVAILABLE = True
except ImportError:
    MMDetection3D_AVAILABLE = False


class FCAF3DDetector:
    """
    FCAF3D detector using MMDetection3D for indoor 3D object detection.

    This detector loads a pre-trained FCAF3D model and performs inference
    on point cloud data, returning CVAT-compatible cuboid annotations.
    """

    def __init__(self, logger=None):
        self.logger = logger
        self.model = None
        self.config = None

        # Initialize MMDetection3D
        if MMDetection3D_AVAILABLE:
            register_all_modules()

        # Default configuration
        self.confidence_threshold = float(os.getenv("DEFAULT_CONFIDENCE_THRESHOLD", "0.3"))

        # Model configuration
        self.config_file = os.getenv("FCAF3D_CONFIG", "configs/fcaf3d/fcaf3d_8x2_scannet-3d-18class.py")
        self.checkpoint_file = os.getenv("FCAF3D_CHECKPOINT", "checkpoints/fcaf3d_scannet.pth")

        self._load_model()

    def _get_device(self):
        """Automatically detect available device (GPU or CPU)."""
        try:
            import torch
            if torch.cuda.is_available():
                # Check for ROCm/CUDA
                if torch.version.hip:
                    if self.logger:
                        self.logger.info("Using ROCm GPU for inference")
                    return 'cuda:0'
                else:
                    if self.logger:
                        self.logger.info("Using CUDA GPU for inference")
                    return 'cuda:0'
            else:
                if self.logger:
                    self.logger.info("Using CPU for inference (GPU not available)")
                return 'cpu'
        except ImportError:
            if self.logger:
                self.logger.warning("PyTorch not available, using CPU")
            return 'cpu'

    def _load_model(self):
        """Load the FCAF3D model and configuration."""
        if not MMDetection3D_AVAILABLE:
            if self.logger:
                self.logger.error("MMDetection3D not available. Please check Docker container setup.")
            return

        try:
            if self.logger:
                self.logger.info(f"Loading FCAF3D model: {self.config_file}")

            # Load config
            self.config = Config.fromfile(self.config_file)

            # Auto-detect device
            device = self._get_device()

            # Initialize the detector
            self.model = init_model(self.config, self.checkpoint_file, device=device)

            if self.logger:
                self.logger.info(f"FCAF3D model loaded successfully on {device}")

        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to load FCAF3D model: {e}")
            raise

    def _convert_to_cvat_format(self, results: Dict[str, Any], threshold: float) -> List[Dict]:
        """
        Convert MMDetection3D results to CVAT cuboid format.

        Args:
            results: MMDetection3D inference results
            threshold: Confidence threshold

        Returns:
            List of CVAT-compatible cuboid detections
        """
        detections = []

        if 'pts_bbox' not in results:
            return detections

        bboxes_3d = results['pts_bbox']  # 3D bounding boxes

        if len(bboxes_3d) == 0:
            return detections

        # Extract predictions
        bbox_preds = bboxes_3d.pred_instances_3d
        scores = bbox_preds.scores_3d.cpu().numpy()
        labels = bbox_preds.labels_3d.cpu().numpy()
        bboxes = bbox_preds.bboxes_3d.cpu().numpy()

        for i, (score, label, bbox) in enumerate(zip(scores, labels, bboxes)):
            if score < threshold:
                continue

            # Convert to CVAT cuboid format
            # FCAF3D outputs in camera coordinates, convert to CVAT format
            center_x, center_y, center_z = bbox[:3]  # center coordinates
            size_x, size_y, size_z = bbox[3:6]       # dimensions
            rot_z = bbox[6] if len(bbox) > 6 else 0.0  # yaw rotation

            detection = {
                "confidence": f"{score:.4f}",
                "label": self._get_label_name(label),
                "type": "cuboid",
                "points": [
                    # Center position (3 values)
                    float(center_x),
                    float(center_y),
                    float(center_z),

                    # Rotation angles (3 values)
                    0.0,  # Rotation X (roll)
                    0.0,  # Rotation Y (pitch)
                    float(rot_z),  # Rotation Z (yaw)

                    # Scale/dimensions (3 values)
                    float(size_x),  # Length (X dimension)
                    float(size_y),  # Width (Y dimension)
                    float(size_z),  # Height (Z dimension)

                    # Pad to 16 points as required by CVAT cuboid validation
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
                ],
            }

            detections.append(detection)

        return detections

    def _get_label_name(self, label_id: int) -> str:
        """Convert label ID to label name."""
        # Map ScanNet class IDs to CVAT labels
        # Focus on pedestrian detection - map person/human classes
        label_map = {
            0: "Pedestrian",  # person
            1: "Pedestrian",  # human
            # Add other mappings as needed
        }
        return label_map.get(label_id, f"class_{label_id}")

    def infer(self, *, cloud_bytes: bytes, threshold: Optional[float] = None, frame_id: int = -1) -> List[Dict]:
        """
        Run inference on point cloud data.

        Args:
            cloud_bytes: Point cloud data as bytes (.bin or .pcd format)
            threshold: Confidence threshold override
            frame_id: Frame number for logging

        Returns:
            List of CVAT-compatible cuboid detections
        """
        if self.model is None:
            if self.logger:
                self.logger.error("Model not loaded")
            return []

        threshold = threshold or self.confidence_threshold

        try:
            # Save point cloud to temporary file
            with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as tmp_file:
                tmp_file.write(cloud_bytes)
                tmp_path = tmp_file.name

            try:
                if self.logger:
                    self.logger.debug(f"Frame {frame_id}: Running FCAF3D inference")

                # Run inference
                results = inference_detector(self.model, tmp_path)

                # Convert to CVAT format
                detections = self._convert_to_cvat_format(results, threshold)

                if self.logger:
                    self.logger.debug(f"Frame {frame_id}: Found {len(detections)} detections")

                return detections

            finally:
                # Clean up temporary file
                os.unlink(tmp_path)

        except Exception as e:
            if self.logger:
                self.logger.error(f"Inference failed for frame {frame_id}: {e}")
            return []


def init_context(context):
    """
    Initialize the FCAF3D detector when Nuclio function starts.
    """
    context.logger.info("Initializing FCAF3D detector (MMDetection3D)")
    context.user_data.detector = FCAF3DDetector(context.logger)
    context.logger.info("FCAF3D detector ready")


def handler(context, event):
    """
    Nuclio handler for processing FCAF3D detection requests.

    CVAT sends requests with:
    - "image": base64-encoded point cloud data
    - "frame": frame number for logging
    - "threshold": optional confidence threshold override

    Returns CVAT-compatible cuboid detections.
    """
    # Parse CVAT request data
    data = event.body
    frame_id = data.get("frame", -1)
    threshold = data.get("threshold")

    # Decode base64 point cloud data
    cloud_bytes = base64.b64decode(data["image"])

    # Run detection
    detections = context.user_data.detector.infer(
        cloud_bytes=cloud_bytes,
        threshold=threshold,
        frame_id=frame_id,
    )

    # Return detections in CVAT-compatible JSON format
    return context.Response(
        body=json.dumps(detections),
        headers={},
        content_type="application/json",
        status_code=200,
    )
