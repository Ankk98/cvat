#!/usr/bin/env python3
"""
Semantic Segmentation & Hand Pose Integration Test
==================================================

Runs semantic segmentation (SAM/Detectron2) and hand pose estimation (MediaPipe)
on the same images, then validates their intersection.

Features:
- Runs SAM or Detectron2 for semantic segmentation
- Runs MediaPipe for hand pose estimation
- Calculates intersection between hand keypoints and semantic masks
- Validates detection consistency
- Generates comprehensive reports

Usage:
    # Test with SAM
    python test_semantic_and_pose_integration.py \
        --dataset-path test-data/egocentric-hands \
        --semantic-model sam \
        --sam-url http://localhost:32800 \
        --mediapipe-url http://localhost:8000

    # Test with Detectron2
    python test_semantic_and_pose_integration.py \
        --dataset-path test-data/egocentric-hands \
        --semantic-model detectron2 \
        --detectron-url http://localhost:32769 \
        --mediapipe-url http://localhost:8000
"""

import argparse
import base64
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import requests
import cv2
import numpy as np
from PIL import Image
import io
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SemanticPoseIntegrationTester:
    """Test integration of semantic segmentation and hand pose estimation."""

    def __init__(self, dataset_path: str,
                 semantic_model: str = "sam",
                 sam_url: Optional[str] = None,
                 detectron_url: Optional[str] = None,
                 mediapipe_url: str = "http://localhost:8000",
                 output_dir: str = "test-results"):
        self.dataset_path = Path(dataset_path)
        self.semantic_model = semantic_model.lower()
        self.sam_url = sam_url.rstrip('/') if sam_url else None
        self.detectron_url = detectron_url.rstrip('/') if detectron_url else None
        self.mediapipe_url = mediapipe_url.rstrip('/')
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Statistics
        self.stats = {
            'total_images': 0,
            'images_with_semantic': 0,
            'images_with_pose': 0,
            'images_with_both': 0,
            'hand_detection_rate': 0.0,
            'intersection_metrics': [],
            'semantic_detection_rate': 0.0,
            'pose_detection_rate': 0.0,
            'consistency_score': 0.0
        }

    def load_images(self) -> List[Path]:
        """Load all images from dataset directory."""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        images = [
            f for f in self.dataset_path.iterdir()
            if f.is_file() and f.suffix.lower() in image_extensions
        ]
        return sorted(images)

    def image_to_base64(self, image_path: Path) -> str:
        """Convert image to base64 string."""
        with open(image_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')

    def call_sam(self, image_b64: str) -> Optional[List[Dict]]:
        """Call SAM service for semantic segmentation."""
        if not self.sam_url:
            logger.warning("SAM URL not provided")
            return None

        try:
            payload = {"image": image_b64}
            response = requests.post(self.sam_url, json=payload, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"SAM service call failed: {e}")
            return None

    def call_detectron2(self, image_b64: str) -> Optional[List[Dict]]:
        """Call Detectron2 service for instance segmentation."""
        if not self.detectron_url:
            logger.warning("Detectron2 URL not provided")
            return None

        try:
            payload = {"image": image_b64, "threshold": 0.3}
            response = requests.post(self.detectron_url, json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()

            # Handle different Detectron2 response formats
            if isinstance(result, list):
                return result
            elif isinstance(result, dict):
                if 'annotations' in result:
                    return result['annotations']
                elif 'detections' in result:
                    return result['detections']
                else:
                    return [result]
            else:
                logger.warning(f"Unexpected Detectron2 response format: {type(result)}")
                return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Detectron2 service call failed: {e}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response status: {e.response.status_code}")
                logger.error(f"Response text: {e.response.text[:200]}")
            return None
        except Exception as e:
            logger.error(f"Detectron2 service error: {e}")
            return None

    def call_mediapipe(self, image_b64: str) -> Optional[List[Dict]]:
        """Call MediaPipe service for hand pose estimation."""
        try:
            payload = {"image": image_b64, "threshold": 0.3}
            response = requests.post(
                f"{self.mediapipe_url}/detect",
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"MediaPipe service call failed: {e}")
            return None

    def extract_semantic_masks(self, semantic_result: List[Dict],
                              image_width: int, image_height: int) -> List[np.ndarray]:
        """Extract semantic segmentation masks from SAM/Detectron2 results."""
        masks = []

        for item in semantic_result:
            mask = None

            # Handle Detectron2 RetinaNet bounding boxes (convert to mask)
            if 'points' in item and item.get('type') == 'rectangle':
                # Bounding box format: [x1, y1, x2, y2]
                points = item['points']
                if len(points) >= 4:
                    x1, y1, x2, y2 = points[0], points[1], points[2], points[3]
                    mask = np.zeros((image_height, image_width), dtype=np.uint8)
                    cv2.rectangle(mask, (int(x1), int(y1)), (int(x2), int(y2)), 1, -1)

            # Handle different mask formats
            elif 'mask' in item:
                # Flattened pixels format (from Mask R-CNN service)
                # Format: [pixel1, pixel2, ..., x_min, y_min, x_max, y_max]
                if isinstance(item['mask'], list) and len(item['mask']) >= 6:
                    mask = self._decode_flattened_pixels(item['mask'], image_width, image_height)
                # Polygon format
                elif 'polygon' in item.get('mask', {}):
                    points = item['mask']['polygon']
                    mask = self._polygon_to_mask(points, image_width, image_height)

            # Handle SAM/other formats (RLE after CVAT backend processing)
            elif 'rle' in item:
                # This is RLE format (after CVAT backend converts flattened pixels)
                mask = self._decode_rle(item['rle'], image_width, image_height)
            elif 'polygon' in item:
                points = item['polygon']
                mask = self._polygon_to_mask(points, image_width, image_height)

            # Handle direct mask array (base64 or numpy)
            elif 'mask_array' in item:
                mask_data = item['mask_array']
                if isinstance(mask_data, str):
                    # Base64 encoded
                    mask_bytes = base64.b64decode(mask_data)
                    mask = np.frombuffer(mask_bytes, dtype=np.uint8).reshape((image_height, image_width))
                else:
                    mask = np.array(mask_data, dtype=np.uint8)

            if mask is not None:
                masks.append(mask)

        return masks

    def extract_hand_keypoints(self, mp_result: List[Dict]) -> Dict[str, Dict]:
        """Extract hand keypoints from MediaPipe results."""
        hand_keypoints = {}

        for skeleton in mp_result:
            if skeleton.get('type') == 'skeleton':
                elements = skeleton.get('elements', [])
                for element in elements:
                    if element.get('type') == 'points':
                        label = element.get('label', '')
                        # Filter for hand-related keypoints
                        if any(term in label.lower() for term in
                              ['hand', 'wrist', 'thumb', 'finger', 'index', 'middle', 'ring', 'pinky']):
                            points = element.get('points', [])
                            if len(points) >= 2:
                                hand_keypoints[label] = {
                                    'x': points[0],
                                    'y': points[1],
                                    'outside': element.get('outside', False),
                                    'confidence': self._get_confidence(element)
                                }

        return hand_keypoints

    def _decode_flattened_pixels(self, flattened: List[int], width: int, height: int) -> np.ndarray:
        """
        Decode flattened pixels format to binary mask.

        Format: [pixel1, pixel2, ..., x_min, y_min, x_max, y_max]
        - Last 4 values are bounding box coordinates
        - Rest are flattened mask pixels (0 or 1) for the tight bounding box
        """
        if len(flattened) < 6:
            raise ValueError(f"Invalid flattened pixels format: need at least 6 values, got {len(flattened)}")

        # Extract bbox and pixel data
        x_min, y_min, x_max, y_max = [int(x) for x in flattened[-4:]]
        pixel_data = flattened[:-4]

        # Calculate tight mask dimensions
        tight_width = x_max - x_min + 1
        tight_height = y_max - y_min + 1

        # Reshape pixel data to tight mask
        if len(pixel_data) != tight_width * tight_height:
            raise ValueError(f"Pixel data length ({len(pixel_data)}) doesn't match bbox size ({tight_width}x{tight_height})")

        tight_mask = np.array(pixel_data, dtype=np.uint8).reshape((tight_height, tight_width))

        # Create full image mask
        full_mask = np.zeros((height, width), dtype=np.uint8)
        full_mask[y_min:y_max+1, x_min:x_max+1] = tight_mask

        return full_mask

    def _decode_rle(self, rle: List[int], width: int, height: int) -> np.ndarray:
        """
        Decode RLE (Run-Length Encoding) to binary mask.

        Note: This is for RLE format (after CVAT backend processing).
        For service output, use _decode_flattened_pixels() instead.
        """
        mask = np.zeros(width * height, dtype=np.uint8)
        for i in range(0, len(rle), 2):
            start = rle[i]
            length = rle[i + 1] if i + 1 < len(rle) else 0
            if start + length <= len(mask):
                mask[start:start + length] = 1
        return mask.reshape((height, width))

    def _polygon_to_mask(self, points: List[float], width: int, height: int) -> np.ndarray:
        """Convert polygon points to binary mask."""
        mask = np.zeros((height, width), dtype=np.uint8)
        if len(points) >= 6:  # At least 3 points (x,y pairs)
            pts = np.array([(points[i], points[i+1]) for i in range(0, len(points), 2)], dtype=np.int32)
            cv2.fillPoly(mask, [pts], 1)
        return mask

    def _get_confidence(self, element: Dict) -> float:
        """Extract confidence from element attributes."""
        attrs = element.get('attributes', [])
        for attr in attrs:
            if attr.get('name') == 'confidence':
                try:
                    return float(attr.get('value', 0.0))
                except:
                    return 0.0
        return 1.0

    def calculate_hand_mask_intersection(self, hand_keypoints: Dict[str, Dict],
                                        semantic_masks: List[np.ndarray],
                                        image_width: int, image_height: int) -> Dict:
        """Calculate intersection of hand keypoints with semantic segmentation masks."""
        if not hand_keypoints or not semantic_masks:
            return {
                'iou': 0.0,
                'intersection_area': 0,
                'union_area': 0,
                'hand_mask_area': 0,
                'semantic_mask_area': 0,
                'overlap_percentage': 0.0
            }

        # Create hand bounding region from keypoints
        hand_points = [(kp['x'], kp['y']) for kp in hand_keypoints.values()
                      if not kp.get('outside', False)]

        if not hand_points:
            return {
                'iou': 0.0,
                'intersection_area': 0,
                'union_area': 0,
                'hand_mask_area': 0,
                'semantic_mask_area': 0,
                'overlap_percentage': 0.0
            }

        x_coords = [p[0] for p in hand_points]
        y_coords = [p[1] for p in hand_points]

        x_min, x_max = max(0, min(x_coords)), min(image_width, max(x_coords))
        y_min, y_max = max(0, min(y_coords)), min(image_height, max(y_coords))

        # Create hand mask (convex hull or bounding box)
        hand_mask = np.zeros((image_height, image_width), dtype=np.uint8)
        if len(hand_points) >= 3:
            pts = np.array(hand_points, dtype=np.int32)
            hull = cv2.convexHull(pts)
            cv2.fillPoly(hand_mask, [hull], 1)
        else:
            cv2.rectangle(hand_mask, (int(x_min), int(y_min)),
                         (int(x_max), int(y_max)), 1, -1)

        hand_mask_area = hand_mask.sum()

        # Calculate IoU with each semantic mask
        best_iou = 0.0
        best_intersection = 0
        best_union = 0
        best_semantic_area = 0

        for sem_mask in semantic_masks:
            # Ensure masks are same size
            if sem_mask.shape != hand_mask.shape:
                sem_mask_resized = cv2.resize(sem_mask.astype(np.uint8),
                                             (image_width, image_height),
                                             interpolation=cv2.INTER_NEAREST)
            else:
                sem_mask_resized = sem_mask.astype(np.uint8)

            semantic_area = sem_mask_resized.sum()
            intersection = np.logical_and(hand_mask, sem_mask_resized).sum()
            union = np.logical_or(hand_mask, sem_mask_resized).sum()
            iou = intersection / union if union > 0 else 0.0

            if iou > best_iou:
                best_iou = iou
                best_intersection = intersection
                best_union = union
                best_semantic_area = semantic_area

        # Calculate overlap percentage (how much of hand region overlaps with semantic mask)
        overlap_percentage = (best_intersection / hand_mask_area * 100) if hand_mask_area > 0 else 0.0

        return {
            'iou': best_iou,
            'intersection_area': int(best_intersection),
            'union_area': int(best_union),
            'hand_mask_area': int(hand_mask_area),
            'semantic_mask_area': int(best_semantic_area),
            'overlap_percentage': overlap_percentage
        }

    def analyze_integration(self, image_path: Path, semantic_result: List[Dict],
                           mp_result: List[Dict], image_width: int, image_height: int) -> Dict:
        """Analyze integration of semantic segmentation and hand pose."""
        analysis = {
            'image': image_path.name,
            'semantic_detections': len(semantic_result) if semantic_result else 0,
            'hand_keypoints': 0,
            'intersection_metrics': {},
            'consistency': {
                'both_detected': False,
                'only_semantic': False,
                'only_pose': False,
                'neither': False
            }
        }

        # Extract semantic masks
        semantic_masks = []
        if semantic_result:
            semantic_masks = self.extract_semantic_masks(semantic_result, image_width, image_height)
            analysis['semantic_masks_count'] = len(semantic_masks)

        # Extract hand keypoints
        hand_keypoints = {}
        if mp_result:
            hand_keypoints = self.extract_hand_keypoints(mp_result)
            analysis['hand_keypoints'] = len(hand_keypoints)

        # Determine consistency
        has_semantic = len(semantic_masks) > 0
        has_pose = len(hand_keypoints) > 0

        if has_semantic and has_pose:
            analysis['consistency']['both_detected'] = True
        elif has_semantic and not has_pose:
            analysis['consistency']['only_semantic'] = True
        elif not has_semantic and has_pose:
            analysis['consistency']['only_pose'] = True
        else:
            analysis['consistency']['neither'] = True

        # Calculate intersection if both exist
        if semantic_masks and hand_keypoints:
            intersection = self.calculate_hand_mask_intersection(
                hand_keypoints, semantic_masks, image_width, image_height
            )
            analysis['intersection_metrics'] = intersection

        return analysis

    def test_dataset(self, max_samples: Optional[int] = None):
        """Test entire dataset."""
        logger.info(f"Testing semantic segmentation & hand pose integration")
        logger.info(f"Dataset: {self.dataset_path}")
        logger.info(f"Semantic model: {self.semantic_model}")

        images = self.load_images()
        if not images:
            logger.error(f"No images found in {self.dataset_path}")
            return

        if max_samples:
            images = images[:max_samples]

        logger.info(f"Found {len(images)} images")

        results = []

        for image_path in images:
            logger.info(f"Processing {image_path.name}...")

            # Load image
            image_b64 = self.image_to_base64(image_path)
            image = Image.open(image_path)
            image_width, image_height = image.size

            # Run semantic segmentation
            semantic_result = None
            has_semantic = False
            if self.semantic_model == 'sam' and self.sam_url:
                semantic_result = self.call_sam(image_b64)
                has_semantic = semantic_result is not None and len(semantic_result) > 0
            elif self.semantic_model == 'detectron2' and self.detectron_url:
                semantic_result = self.call_detectron2(image_b64)
                has_semantic = semantic_result is not None and len(semantic_result) > 0
            else:
                logger.debug(f"Skipping semantic segmentation (service not available)")

            # Run MediaPipe hand pose
            mp_result = self.call_mediapipe(image_b64)
            has_pose = mp_result is not None and len(mp_result) > 0

            # Update statistics
            self.stats['total_images'] += 1
            if has_semantic:
                self.stats['images_with_semantic'] += 1
            if has_pose:
                self.stats['images_with_pose'] += 1
            if has_semantic and has_pose:
                self.stats['images_with_both'] += 1
                self.stats['hand_detection_rate'] += 1

            # Analyze integration
            if semantic_result is not None and mp_result is not None:
                analysis = self.analyze_integration(
                    image_path, semantic_result, mp_result, image_width, image_height
                )
                results.append(analysis)

                # Update statistics
                if analysis['intersection_metrics']:
                    self.stats['intersection_metrics'].append(analysis['intersection_metrics'])

        # Calculate final statistics
        if self.stats['total_images'] > 0:
            self.stats['semantic_detection_rate'] = self.stats['images_with_semantic'] / self.stats['total_images']
            self.stats['pose_detection_rate'] = self.stats['images_with_pose'] / self.stats['total_images']
            self.stats['hand_detection_rate'] = self.stats['hand_detection_rate'] / self.stats['total_images']
            self.stats['consistency_score'] = self.stats['images_with_both'] / self.stats['total_images']

        # Save results
        results_path = self.output_dir / f"semantic_pose_integration_{self.semantic_model}_{int(time.time())}.json"
        with open(results_path, 'w') as f:
            json.dump({
                'statistics': self.stats,
                'detailed_results': results,
                'dataset_path': str(self.dataset_path),
                'semantic_model': self.semantic_model,
                'sam_url': self.sam_url,
                'detectron_url': self.detectron_url,
                'mediapipe_url': self.mediapipe_url
            }, f, indent=2)

        logger.info(f"Analysis complete. Results saved to {results_path}")
        self.print_summary()

    def print_summary(self):
        """Print analysis summary."""
        print("\n" + "="*80)
        print("SEMANTIC SEGMENTATION & HAND POSE INTEGRATION SUMMARY")
        print("="*80)
        print(f"Total images analyzed: {self.stats['total_images']}")
        print(f"Semantic model: {self.semantic_model.upper()}")
        print(f"\nDetection Rates:")
        print(f"  Semantic segmentation: {self.stats['semantic_detection_rate']:.2%}")
        print(f"  Hand pose estimation: {self.stats['pose_detection_rate']:.2%}")
        print(f"  Both detected: {self.stats['consistency_score']:.2%}")

        if self.stats['intersection_metrics']:
            avg_iou = np.mean([m['iou'] for m in self.stats['intersection_metrics']])
            avg_overlap = np.mean([m['overlap_percentage'] for m in self.stats['intersection_metrics']])
            print(f"\nIntersection Metrics:")
            print(f"  Average IoU: {avg_iou:.3f}")
            print(f"  Average overlap: {avg_overlap:.1f}%")
            print(f"  Images with intersection: {len(self.stats['intersection_metrics'])}")

        print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Test integration of semantic segmentation and hand pose estimation"
    )
    parser.add_argument('--dataset-path', required=True,
                       help='Path to dataset directory with images')
    parser.add_argument('--semantic-model', choices=['sam', 'detectron2'], default='sam',
                       help='Semantic segmentation model to use')
    parser.add_argument('--sam-url', default=None,
                       help='SAM service URL (e.g., http://localhost:32800)')
    parser.add_argument('--detectron-url', default=None,
                       help='Detectron2 service URL (e.g., http://localhost:32769)')
    parser.add_argument('--mediapipe-url', default='http://localhost:8000',
                       help='MediaPipe service URL')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Maximum number of images to test')
    parser.add_argument('--output-dir', default='test-results',
                       help='Output directory for results')

    args = parser.parse_args()

    # Auto-detect service URLs if not provided
    if args.semantic_model == 'sam' and not args.sam_url:
        # Try common SAM ports
        for port in [32800, 32768, 32770]:
            try:
                response = requests.get(f"http://localhost:{port}/health", timeout=2)
                if response.status_code == 200:
                    args.sam_url = f"http://localhost:{port}"
                    logger.info(f"Auto-detected SAM URL: {args.sam_url}")
                    break
            except:
                continue
        if not args.sam_url:
            logger.warning("SAM service not found. Will test MediaPipe only.")

    if args.semantic_model == 'detectron2' and not args.detectron_url:
        # Try common Detectron2 ports
        for port in [32769, 32770]:
            try:
                response = requests.get(f"http://localhost:{port}/health", timeout=2)
                if response.status_code == 200:
                    args.detectron_url = f"http://localhost:{port}"
                    logger.info(f"Auto-detected Detectron2 URL: {args.detectron_url}")
                    break
            except:
                continue
        if not args.detectron_url:
            logger.warning("Detectron2 service not found. Will test MediaPipe only.")

    tester = SemanticPoseIntegrationTester(
        dataset_path=args.dataset_path,
        semantic_model=args.semantic_model,
        sam_url=args.sam_url,
        detectron_url=args.detectron_url,
        mediapipe_url=args.mediapipe_url,
        output_dir=args.output_dir
    )

    tester.test_dataset(args.max_samples)


if __name__ == '__main__':
    main()

