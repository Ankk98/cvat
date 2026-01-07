import base64
import json
import tempfile
import os
import sys
import numpy as np
from typing import List, Dict, Any, Optional

# Monkey-patch mmcv.ops.nms3d_normal BEFORE importing mmdet3d
# This ensures that when mmdet3d modules import it, they get the wrapper
try:
    import torch
    import mmcv.ops.iou3d as iou3d_ops
    import mmcv.ops

    # Save original function
    if not hasattr(iou3d_ops, '_original_nms3d_normal'):
        iou3d_ops._original_nms3d_normal = iou3d_ops.nms3d_normal

    def nms3d_normal_wrapper(boxes, scores, iou_threshold):
        """
        Wrapper to move tensors to GPU for NMS, then back to CPU.
        Needed because FCAF3D backbone runs on CPU (MinkowskiEngine requirement)
        but NMS requires GPU (MMCV requirement).
        """
        is_cpu = boxes.device.type == 'cpu'
        if is_cpu and torch.cuda.is_available():
            # Move to GPU
            boxes_gpu = boxes.cuda()
            scores_gpu = scores.cuda()

            # Run NMS on GPU
            inds = iou3d_ops._original_nms3d_normal(boxes_gpu, scores_gpu, iou_threshold)

            # Move result back to CPU
            return inds.cpu()
        else:
            return iou3d_ops._original_nms3d_normal(boxes, scores, iou_threshold)

    # Apply patch to both locations where it might be imported
    iou3d_ops.nms3d_normal = nms3d_normal_wrapper
    mmcv.ops.nms3d_normal = nms3d_normal_wrapper
    print("✅ Patched mmcv.ops.nms3d_normal for hybrid CPU-GPU execution")

except ImportError:
    # mmcv not installed on host, ignore
    pass
except Exception as e:
    print(f"⚠️ Failed to patch NMS: {e}")

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
        self.config_file = os.getenv("FCAF3D_CONFIG", "configs/fcaf3d/fcaf3d_2xb8_scannet-3d-18class.py")
        self.checkpoint_file = os.getenv("FCAF3D_CHECKPOINT", "checkpoints/fcaf3d_8x2_scannet-3d-18class_20220805_084956.pth")

        if self.logger:
            self.logger.info(f"Config file: {self.config_file}")
            self.logger.info(f"Checkpoint file: {self.checkpoint_file}")
            self.logger.info("Starting model loading...")

        self._load_model()

    def _get_device(self):
        """Automatically detect available device (GPU or CPU)."""
        import torch
        # Force CPU for FCAF3D models due to MinkowskiEngine CPU-only compatibility
        # We handle the NMS GPU requirement via monkey-patching
        if self.logger:
            self.logger.info("🎯 Selected: CPU for FCAF3D inference (MinkowskiEngine CPU-only compatibility)")
            if not torch.cuda.is_available():
                self.logger.warning("⚠️ JDBC: GPU not available - NMS might fail if no CPU implementation exists")
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
        data_sample = None

        if isinstance(results, dict):
            if 'pts_bbox' in results:
                data_sample = results['pts_bbox']
            else:
                if self.logger:
                    self.logger.warning("⚠️ No 'pts_bbox' key found in results dictionary")
                return detections
        elif isinstance(results, (list, tuple)):
            if len(results) > 0:
                data_sample = results[0]
            else:
                if self.logger:
                    self.logger.warning("⚠️ Results list/tuple is empty")
                return detections
        else:
            # Assume results is the DetDataSample itself
            data_sample = results

        if self.logger:
            self.logger.info(f"Using data sample for conversion: {type(data_sample)}")

        # Extract predictions
        try:
            if hasattr(data_sample, 'pred_instances_3d'):
                bbox_preds = data_sample.pred_instances_3d
                if self.logger:
                    self.logger.info("📦 Detected MMDetection3D 1.x pred_instances_3d format")

                scores = bbox_preds.scores_3d.cpu().numpy()
                labels = bbox_preds.labels_3d.cpu().numpy()
                # Newer mmengine/mmdet3d uses .tensor attribute for bboxes
                if hasattr(bbox_preds.bboxes_3d, 'tensor'):
                    bboxes = bbox_preds.bboxes_3d.tensor.cpu().numpy()
                else:
                    bboxes = bbox_preds.bboxes_3d.cpu().numpy()
            elif hasattr(data_sample, 'bboxes_3d'):
                if self.logger:
                    self.logger.info("📦 Detected direct bbox object format")
                scores = data_sample.scores_3d.cpu().numpy() if hasattr(data_sample, 'scores_3d') else np.array([])
                labels = data_sample.labels_3d.cpu().numpy() if hasattr(data_sample, 'labels_3d') else np.array([])
                bboxes = data_sample.bboxes_3d.tensor.cpu().numpy() if hasattr(data_sample.bboxes_3d, 'tensor') else data_sample.bboxes_3d.cpu().numpy()
            else:
                if self.logger:
                    self.logger.error(f"❌ Could not find predictions in data sample type {type(data_sample)}")
                return detections
        except Exception as e:
            if self.logger:
                self.logger.error(f"❌ Error extracting predictions: {e}")
            return detections

        if self.logger:
            self.logger.info(f"Raw predictions: {len(scores)} detections")
            if len(scores) > 0:
                self.logger.info(f"Score range: {scores.min():.4f} - {scores.max():.4f}")
                self.logger.info(f"Label range: {labels.min()} - {labels.max()}")
                self.logger.info(f"Bbox shape: {bboxes.shape}")
            else:
                self.logger.info("ℹ️ No detections found in predictions")
                return detections

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

    def _pcd_to_bin(self, pcd_bytes: bytes) -> np.ndarray:
        """
        Convert PCD format to FCAF3D format for MMDetection3D compatibility.

        Args:
            pcd_bytes: PCD file data as bytes

        Returns:
            np.ndarray: Point cloud in FCAF3D format (N, 6) [x, y, z, r, g, b]
        """
        import io

        if self.logger:
            self.logger.debug("Converting PCD to BIN format...")

        # Save PCD to temporary file for Open3D to read
        with tempfile.NamedTemporaryFile(suffix='.pcd', delete=False) as pcd_tmp:
            pcd_tmp.write(pcd_bytes)
            pcd_path = pcd_tmp.name

        try:
            # Use Open3D to read PCD file (same as SIT detector)
            try:
                import open3d as o3d
            except ImportError:
                if self.logger:
                    self.logger.error("❌ Open3D not available for PCD conversion")
                raise

            pcd = o3d.io.read_point_cloud(pcd_path)
            points = np.asarray(pcd.points)

            if self.logger:
                self.logger.debug(f"Read {len(points)} points from PCD")

            # Convert to FCAF3D format: [x, y, z, r, g, b] (6 values per point)
            # FCAF3D uses load_dim=6 for ScanNet dataset with RGB color information
            if points.shape[1] == 3:
                # Add RGB channels (set to 128 for all points = neutral gray)
                rgb = np.full((len(points), 3), 128, dtype=np.float32)
                points_bin = np.concatenate([points, rgb], axis=1).astype(np.float32)
            else:
                # If already has more than 3 channels, ensure exactly 6
                points_bin = points[:, :6].astype(np.float32)
                if points.shape[1] < 6:
                    # Pad with zeros if needed
                    padding = np.zeros((len(points), 6 - points.shape[1]), dtype=np.float32)
                    points_bin = np.concatenate([points_bin, padding], axis=1)

            if self.logger:
                self.logger.debug(f"Converted to BIN format: {points_bin.shape}")

            return points_bin

        finally:
            # Clean up temporary PCD file
            os.unlink(pcd_path)

    def _get_label_name(self, label_id: int) -> str:
        """Convert label ID to ScanNet label name."""
        # Map ScanNet class IDs to standard names
        label_map = {
            0: "cabinet",
            1: "bed",
            2: "chair",
            3: "sofa",
            4: "table",
            5: "door",
            6: "window",
            7: "bookshelf",
            8: "picture",
            9: "counter",
            10: "desk",
            11: "curtain",
            12: "refrigerator",
            13: "showercurtain",
            14: "toilet",
            15: "sink",
            16: "bathtub",
            17: "otherfurniture"
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
            # Detect format and prepare point cloud file
            import sys
            print(f"DEBUG: FCAF3D infer called with {len(cloud_bytes)} bytes", file=sys.stderr)
            if self.logger:
                self.logger.info("💾 Preparing point cloud file...")
                self.logger.info(f"🔍 DEBUG: Received {len(cloud_bytes)} bytes of data")

            # Detect format using same logic as SIT detector
            header = cloud_bytes[:16]
            if self.logger:
                self.logger.info(f"🔍 File header: {header[:20]!r}")

            is_pcd = header.startswith(b"VERSION") or header.startswith(b"# .PCD")

            if is_pcd:
                if self.logger:
                    self.logger.info("📄 Detected PCD format - converting to BIN for MMDetection3D")
                try:
                    # Convert PCD to BIN format for MMDetection3D compatibility
                    points = self._pcd_to_bin(cloud_bytes)
                    file_suffix = '.bin'
                    file_data = points.tobytes()
                    if self.logger:
                        self.logger.info(f"📊 Converted {len(points)} points ({len(file_data)} bytes) to BIN format")
                except Exception as e:
                    if self.logger:
                        self.logger.warning(f"❌ PCD conversion failed: {e}, falling back to raw data")
                    # Fallback: use raw data
                    file_suffix = '.bin'
                    file_data = cloud_bytes
            else:
                if len(cloud_bytes) % 4 == 0 and len(cloud_bytes) >= 16:
                    if self.logger:
                        self.logger.info("📄 Assuming BIN format - using as-is")
                    file_suffix = '.bin'
                    file_data = cloud_bytes

                    # DEBUG: Analyze point cloud statistics
                    try:
                        points_debug = np.frombuffer(cloud_bytes, dtype=np.float32).reshape(-1, 4) if len(cloud_bytes) % 16 == 0 else np.frombuffer(cloud_bytes, dtype=np.float32).reshape(-1, 6)
                        if self.logger:
                            x, y, z = points_debug[:, 0], points_debug[:, 1], points_debug[:, 2]
                            self.logger.info(f"🔍 Point Cloud Stats (Frame {frame_id}):")
                            self.logger.info(f"   Count: {len(points_debug)}")
                            self.logger.info(f"   X range: [{x.min():.2f}, {x.max():.2f}] (Span: {x.max()-x.min():.2f})")
                            self.logger.info(f"   Y range: [{y.min():.2f}, {y.max():.2f}] (Span: {y.max()-y.min():.2f})")
                            self.logger.info(f"   Z range: [{z.min():.2f}, {z.max():.2f}] (Span: {z.max()-z.min():.2f})")
                            self.logger.info(f"   Scale Check: If spans are big (>100), inputs might be millimeters. If small (<10), likely meters.")
                    except Exception as e:
                        if self.logger:
                            self.logger.warning(f"⚠️ Could not analyze point stats: {e}")

                    if self.logger:
                        self.logger.info(f"📊 Using BIN data: {len(file_data)} bytes")
                else:
                    # Data doesn't look like BIN, try PCD conversion as fallback
                    if self.logger:
                        self.logger.info("📄 Data doesn't look like BIN, trying PCD conversion")
                    try:
                        points = self._pcd_to_bin(cloud_bytes)
                        file_suffix = '.bin'
                        file_data = points.tobytes()
                        if self.logger:
                            self.logger.info(f"📊 PCD fallback: {len(points)} points ({len(file_data)} bytes) to BIN format")
                    except Exception as e:
                        if self.logger:
                            self.logger.error(f"❌ PCD fallback failed: {e}, using data as-is")
                        file_suffix = '.bin'
                        file_data = cloud_bytes

            with tempfile.NamedTemporaryFile(suffix=file_suffix, delete=False) as tmp_file:
                tmp_file.write(file_data)
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
                    self.logger.info("✅ FCAF3D inference completed")
                    self.logger.info(f"⏱️ Inference time: {end_time - start_time:.3f}s" if start_time else "N/A")
                    if isinstance(results, dict):
                        self.logger.info(f"📊 Raw results keys: {list(results.keys())}")
                    elif isinstance(results, (list, tuple)):
                        self.logger.info(f"📊 Raw results is {type(results)} of length {len(results)}")
                        if len(results) > 0:
                            self.logger.info(f"📊 First element type: {type(results[0])}")
                            self.logger.info(f"📊 First element attributes: {[attr for attr in dir(results[0]) if not attr.startswith('_')][:10]}")
                    else:
                        self.logger.info(f"📊 Raw results type: {type(results)}")
                        self.logger.info(f"📊 Attributes: {[attr for attr in dir(results) if not attr.startswith('_')][:10]}")

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
            raise e


def init_context(context):
    """
    Initialize the FCAF3D detector when Nuclio function starts.
    """
    context.logger.info("🚀 === NUCLIO FUNCTION INITIALIZATION ===")
    context.logger.info("Initializing FCAF3D detector (MMDetection3D)")
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
# Force rebuild 2
