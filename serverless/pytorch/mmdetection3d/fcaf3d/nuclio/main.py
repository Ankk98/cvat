import base64
import json
import tempfile
import os
import sys
from typing import List, Dict, Any, Optional

# MMDetection3D imports - PYTHONPATH should be set correctly by Nuclio
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

        if self.logger:
            self.logger.info("=== FCAF3D Detector Initialization ===")
            self.logger.info(f"MMDetection3D available: {MMDetection3D_AVAILABLE}")
            self.logger.info(f"Python version: {sys.version}")
            self.logger.info(f"Current working directory: {os.getcwd()}")
            self.logger.info(f"PYTHONPATH: {os.environ.get('PYTHONPATH', 'NOT SET')}")

        # Initialize MMDetection3D
        if MMDetection3D_AVAILABLE:
            if self.logger:
                self.logger.info("Registering all MMDetection3D modules...")
            register_all_modules()
            if self.logger:
                self.logger.info("MMDetection3D modules registered successfully")
        else:
            if self.logger:
                self.logger.error("MMDetection3D not available - model loading will fail")

        # Default configuration
        self.confidence_threshold = float(os.getenv("DEFAULT_CONFIDENCE_THRESHOLD", "0.3"))
        if self.logger:
            self.logger.info(f"Default confidence threshold: {self.confidence_threshold}")

        # Model configuration
        self.config_file = os.getenv("FCAF3D_CONFIG", "configs/fcaf3d/fcaf3d_8x2_scannet-3d-18class.py")
        self.checkpoint_file = os.getenv("FCAF3D_CHECKPOINT", "checkpoints/fcaf3d_scannet.pth")

        if self.logger:
            self.logger.info(f"Config file: {self.config_file}")
            self.logger.info(f"Checkpoint file: {self.checkpoint_file}")
            self.logger.info("Starting model loading...")

        self._load_model()

    def _get_device(self):
        """Automatically detect available device (GPU or CPU)."""
        if self.logger:
            self.logger.info("🔍 Device Detection:")

        try:
            import torch
            if self.logger:
                self.logger.info(f"PyTorch version: {torch.__version__}")
                self.logger.info(f"CUDA available: {torch.cuda.is_available()}")
                self.logger.info(f"ROCm/HIP available: {hasattr(torch.version, 'hip') and torch.version.hip is not None}")
                if torch.cuda.is_available():
                    self.logger.info(f"GPU count: {torch.cuda.device_count()}")
                    for i in range(torch.cuda.device_count()):
                        self.logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")

            if torch.cuda.is_available():
                # Check for ROCm/CUDA
                if hasattr(torch.version, 'hip') and torch.version.hip:
                    if self.logger:
                        self.logger.info("🎯 Selected: ROCm GPU for inference (cuda:0)")
                    return 'cuda:0'
                else:
                    if self.logger:
                        self.logger.info("🎯 Selected: CUDA GPU for inference (cuda:0)")
                    return 'cuda:0'
            else:
                if self.logger:
                    self.logger.info("🎯 Selected: CPU for inference (GPU not available)")
                return 'cpu'
        except ImportError as e:
            if self.logger:
                self.logger.warning(f"⚠️ PyTorch not available, using CPU: {e}")
            return 'cpu'

    def _load_model(self):
        """Load the FCAF3D model and configuration."""
        if self.logger:
            self.logger.info("=== Model Loading Process ===")

        if not MMDetection3D_AVAILABLE:
            if self.logger:
                self.logger.error("❌ CRITICAL: MMDetection3D not available. Please check Docker container setup.")
                self.logger.error("Available modules check:")
                try:
                    import sys
                    self.logger.error(f"Python path: {sys.path[:3]}")
                    import mmcv
                    self.logger.error("mmcv can be imported")
                except ImportError as ie:
                    self.logger.error(f"mmcv import failed: {ie}")
            return

        try:
            if self.logger:
                self.logger.info(f"📁 Loading config from: {self.config_file}")

            # Check if config file exists
            if not os.path.exists(self.config_file):
                if self.logger:
                    self.logger.error(f"❌ Config file not found: {self.config_file}")
                    self.logger.error(f"Current directory contents: {os.listdir('.') if os.path.exists('.') else 'N/A'}")
                raise FileNotFoundError(f"Config file not found: {self.config_file}")

            # Load config
            if self.logger:
                self.logger.info("🔧 Loading configuration file...")
            self.config = Config.fromfile(self.config_file)
            if self.logger:
                self.logger.info("✅ Config loaded successfully")
                self.logger.info(f"Config keys: {list(self.config.keys()) if hasattr(self.config, 'keys') else 'N/A'}")

            # Auto-detect device
            if self.logger:
                self.logger.info("🔍 Detecting available device...")
            device = self._get_device()
            if self.logger:
                self.logger.info(f"🎯 Selected device: {device}")

            # Check if checkpoint file exists
            if not os.path.exists(self.checkpoint_file):
                if self.logger:
                    self.logger.error(f"❌ Checkpoint file not found: {self.checkpoint_file}")
                    self.logger.error(f"Current directory contents: {os.listdir('.') if os.path.exists('.') else 'N/A'}")
                raise FileNotFoundError(f"Checkpoint file not found: {self.checkpoint_file}")

            # Initialize the detector
            if self.logger:
                self.logger.info("🚀 Initializing FCAF3D model...")
                self.logger.info(f"Config file: {self.config_file}")
                self.logger.info(f"Checkpoint file: {self.checkpoint_file}")
                self.logger.info(f"Device: {device}")

            self.model = init_model(self.config, self.checkpoint_file, device=device)

            if self.logger:
                self.logger.info("✅ SUCCESS: FCAF3D model loaded successfully!")
                self.logger.info(f"📊 Model device: {device}")
                self.logger.info(f"🏗️ Model type: {type(self.model).__name__}")

                # Log model details
                try:
                    if hasattr(self.model, 'bbox_head') and self.model.bbox_head:
                        if hasattr(self.model.bbox_head, 'num_classes'):
                            self.logger.info(f"📋 Number of classes: {self.model.bbox_head.num_classes}")
                    if hasattr(self.model, 'backbone'):
                        self.logger.info(f"🔧 Backbone type: {type(self.model.backbone).__name__}")
                except Exception as e:
                    self.logger.warning(f"Could not extract model details: {e}")

        except Exception as e:
            if self.logger:
                self.logger.error(f"❌ CRITICAL: Failed to load FCAF3D model: {e}")
                import traceback
                self.logger.error(f"Full traceback: {traceback.format_exc()}")
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
        if self.logger:
            self.logger.info(f"🔄 Converting results to CVAT format (threshold: {threshold})")
            self.logger.info(f"Input results type: {type(results)}")
            if isinstance(results, dict):
                self.logger.info(f"Results keys: {list(results.keys())}")

        detections = []

        if 'pts_bbox' not in results:
            if self.logger:
                self.logger.warning("⚠️ No 'pts_bbox' key found in results")
            return detections

        bboxes_3d = results['pts_bbox']  # 3D bounding boxes
        if self.logger:
            self.logger.info(f"3D bbox object type: {type(bboxes_3d)}")

        if len(bboxes_3d) == 0:
            if self.logger:
                self.logger.info("ℹ️ No 3D bounding boxes found")
            return detections

        # Extract predictions
        bbox_preds = bboxes_3d.pred_instances_3d
        if self.logger:
            self.logger.info(f"Bbox predictions type: {type(bbox_preds)}")

        scores = bbox_preds.scores_3d.cpu().numpy()
        labels = bbox_preds.labels_3d.cpu().numpy()
        bboxes = bbox_preds.bboxes_3d.cpu().numpy()

        if self.logger:
            self.logger.info(f"Raw predictions: {len(scores)} detections")
            self.logger.info(f"Score range: {scores.min():.4f} - {scores.max():.4f}")
            self.logger.info(f"Label range: {labels.min()} - {labels.max()}")
            self.logger.info(f"Bbox shape: {bboxes.shape}")

        processed_count = 0
        filtered_count = 0

        for i, (score, label, bbox) in enumerate(zip(scores, labels, bboxes)):
            processed_count += 1

            if score < threshold:
                filtered_count += 1
                continue

            if self.logger and i < 3:  # Log first few detections
                self.logger.info(f"Processing detection {i+1}: score={score:.4f}, label={label}, bbox={bbox}")

            # Convert to CVAT cuboid format
            # FCAF3D outputs in camera coordinates, convert to CVAT format
            center_x, center_y, center_z = bbox[:3]  # center coordinates
            size_x, size_y, size_z = bbox[3:6]       # dimensions
            rot_z = bbox[6] if len(bbox) > 6 else 0.0  # yaw rotation

            label_name = self._get_label_name(label)

            detection = {
                "confidence": f"{score:.4f}",
                "label": label_name,
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

        if self.logger:
            self.logger.info(f"📊 Conversion summary:")
            self.logger.info(f"  Total raw detections: {processed_count}")
            self.logger.info(f"  Filtered by threshold: {filtered_count}")
            self.logger.info(f"  Final CVAT detections: {len(detections)}")
            if detections:
                self.logger.info(f"  Sample detection: {detections[0]['label']} (conf: {detections[0]['confidence']})")

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
        if self.logger:
            self.logger.info(f"=== FCAF3D Inference Frame {frame_id} ===")
            self.logger.info(f"Input data size: {len(cloud_bytes)} bytes")
            self.logger.info(f"Using threshold: {threshold or self.confidence_threshold}")

        if self.model is None:
            if self.logger:
                self.logger.error("❌ CRITICAL: Model not loaded - cannot perform inference")
            return []

        threshold = threshold or self.confidence_threshold

        try:
            # Save point cloud to temporary file
            if self.logger:
                self.logger.info("💾 Creating temporary point cloud file...")
            with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as tmp_file:
                tmp_file.write(cloud_bytes)
                tmp_path = tmp_file.name

            if self.logger:
                self.logger.info(f"📁 Temporary file created: {tmp_path}")

            try:
                if self.logger:
                    self.logger.info(f"🚀 Running FCAF3D inference on frame {frame_id}...")

                # Run inference
                start_time = os.times()[4] if hasattr(os, 'times') else 0
                results = inference_detector(self.model, tmp_path)
                end_time = os.times()[4] if hasattr(os, 'times') else 0

                if self.logger:
                    self.logger.info("✅ FCAF3D inference completed"                    self.logger.info(f"⏱️ Inference time: {end_time - start_time:.3f}s" if start_time else "N/A")
                    self.logger.info(f"📊 Raw results keys: {list(results.keys()) if isinstance(results, dict) else type(results)}")

                # Convert to CVAT format
                if self.logger:
                    self.logger.info("🔄 Converting results to CVAT format...")
                detections = self._convert_to_cvat_format(results, threshold)

                if self.logger:
                    self.logger.info(f"✅ Frame {frame_id}: Found {len(detections)} detections above threshold {threshold}")
                    for i, det in enumerate(detections[:3]):  # Log first 3 detections
                        self.logger.info(f"  Detection {i+1}: {det['label']} (conf: {det['confidence']})")

                return detections

            finally:
                # Clean up temporary file
                if self.logger:
                    self.logger.info("🧹 Cleaning up temporary file...")
                os.unlink(tmp_path)

        except Exception as e:
            if self.logger:
                self.logger.error(f"❌ CRITICAL: Inference failed for frame {frame_id}: {e}")
                import traceback
                self.logger.error(f"Full traceback:\n{traceback.format_exc()}")
            return []


def init_context(context):
    """
    Initialize the FCAF3D detector when Nuclio function starts.
    """
    context.logger.info("🚀 === NUCLIO FUNCTION INITIALIZATION ===")
    context.logger.info("Initializing FCAF3D detector (MMDetection3D)")
    context.logger.info(f"Function name: {context.name}")
    context.logger.info(f"Python version: {sys.version}")

    # Environment info
    context.logger.info("📋 Environment:")
    context.logger.info(f"  PYTHONPATH: {os.environ.get('PYTHONPATH', 'NOT SET')}")
    context.logger.info(f"  CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'NOT SET')}")
    context.logger.info(f"  FCAF3D_CONFIG: {os.environ.get('FCAF3D_CONFIG', 'NOT SET')}")
    context.logger.info(f"  FCAF3D_CHECKPOINT: {os.environ.get('FCAF3D_CHECKPOINT', 'NOT SET')}")

    try:
        context.logger.info("🏗️ Creating FCAF3D detector instance...")
        context.user_data.detector = FCAF3DDetector(context.logger)
        context.logger.info("✅ FCAF3D detector ready for inference!")
    except Exception as e:
        context.logger.error(f"❌ CRITICAL: Failed to initialize FCAF3D detector: {e}")
        import traceback
        context.logger.error(f"Full initialization error:\n{traceback.format_exc()}")
        raise


def handler(context, event):
    """
    Nuclio handler for processing FCAF3D detection requests.

    CVAT sends requests with:
    - "image": base64-encoded point cloud data
    - "frame": frame number for logging
    - "threshold": optional confidence threshold override

    Returns CVAT-compatible cuboid detections.
    """
    start_time = os.times()[4] if hasattr(os, 'times') else 0

    context.logger.info(f"=== FCAF3D REQUEST HANDLER ===")

    try:
        # Parse CVAT request data
        context.logger.info("📨 Parsing request data...")
        data = event.body
        frame_id = data.get("frame", -1)
        threshold = data.get("threshold")

        context.logger.info(f"🎯 Frame ID: {frame_id}")
        context.logger.info(f"🎚️ Threshold: {threshold if threshold is not None else 'default'}")

        # Decode base64 point cloud data
        context.logger.info("🔓 Decoding base64 point cloud data...")
        cloud_bytes = base64.b64decode(data["image"])
        context.logger.info(f"📊 Point cloud size: {len(cloud_bytes)} bytes")

        # Run detection
        context.logger.info("🎯 Starting FCAF3D detection...")
        detections = context.user_data.detector.infer(
            cloud_bytes=cloud_bytes,
            threshold=threshold,
            frame_id=frame_id,
        )

        end_time = os.times()[4] if hasattr(os, 'times') else 0

        # Log results
        context.logger.info(f"✅ Detection completed!")
        context.logger.info(f"📈 Total time: {end_time - start_time:.3f}s" if start_time else "N/A")
        context.logger.info(f"🔍 Detections found: {len(detections)}")

        # Return detections in CVAT-compatible JSON format
        response_body = json.dumps(detections)
        context.logger.info(f"📤 Response size: {len(response_body)} characters")

        return context.Response(
            body=response_body,
            headers={},
            content_type="application/json",
            status_code=200,
        )

    except Exception as e:
        context.logger.error(f"❌ CRITICAL: Handler failed: {e}")
        import traceback
        context.logger.error(f"Full handler error:\n{traceback.format_exc()}")

        # Return error response
        return context.Response(
            body=json.dumps({"error": str(e)}),
            headers={},
            content_type="application/json",
            status_code=500,
        )
