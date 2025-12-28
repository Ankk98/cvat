#!/usr/bin/env python3
"""
MediaPipe Ground Truth Validation Test (CVAT-Independent)
==========================================================

Tests MediaPipe hand pose estimation against ground truth annotations from local files.
Validates intersection of semantic segmentation & hand pose estimation.

Features:
- Loads ground truth annotations from local JSON/XML files
- Compares MediaPipe predictions with ground truth
- Calculates keypoint accuracy (distance, visibility)
- Validates hand detection rate
- Checks intersection with semantic segmentation masks (from SAM/Detectron2)
- Generates comprehensive accuracy reports

Usage:
    # Test with local annotations
    python test_mediapipe_ground_truth.py \
        --dataset-path test-data/egocentric-hands \
        --annotations annotations.json \
        --mediapipe-url http://localhost:8000

    # Test with semantic segmentation intersection
    python test_mediapipe_ground_truth.py \
        --dataset-path test-data/egocentric-hands \
        --annotations annotations.json \
        --semantic-masks sam_results.json \
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
import xml.etree.ElementTree as ET

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MediaPipeGroundTruthTester:
    """Test MediaPipe against ground truth annotations from local files."""

    def __init__(self, dataset_path: str, annotations_path: Optional[str] = None,
                 semantic_masks_path: Optional[str] = None,
                 mediapipe_url: str = "http://localhost:8000",
                 output_dir: str = "test-results"):
        self.dataset_path = Path(dataset_path)
        self.annotations_path = Path(annotations_path) if annotations_path else None
        self.semantic_masks_path = Path(semantic_masks_path) if semantic_masks_path else None
        self.mediapipe_url = mediapipe_url.rstrip('/')
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Load ground truth annotations
        self.gt_annotations = {}
        if self.annotations_path and self.annotations_path.exists():
            self.gt_annotations = self._load_annotations(self.annotations_path)

        # Load semantic segmentation masks
        self.semantic_masks = {}
        if self.semantic_masks_path and self.semantic_masks_path.exists():
            self.semantic_masks = self._load_semantic_masks(self.semantic_masks_path)

        # Statistics
        self.stats = {
            'total_images': 0,
            'images_with_gt': 0,
            'images_with_mp': 0,
            'images_with_both': 0,
            'hand_detection_rate': 0.0,
            'keypoint_accuracy': [],
            'keypoint_distance_errors': [],
            'visibility_accuracy': [],
            'intersection_metrics': []
        }

    def _load_annotations(self, annotations_path: Path) -> Dict:
        """Load ground truth annotations from JSON or XML file."""
        if annotations_path.suffix == '.json':
            with open(annotations_path, 'r') as f:
                return json.load(f)
        elif annotations_path.suffix == '.xml':
            return self._load_cvat_xml(annotations_path)
        else:
            logger.warning(f"Unsupported annotation format: {annotations_path.suffix}")
            return {}

    def _load_cvat_xml(self, xml_path: Path) -> Dict:
        """Load CVAT XML annotations."""
        tree = ET.parse(xml_path)
        root = tree.getroot()
        annotations = {}

        for image_elem in root.findall('image'):
            image_name = image_elem.get('name')
            annotations[image_name] = {
                'skeletons': [],
                'masks': []
            }

            # Load skeleton annotations
            for skeleton_elem in image_elem.findall('skeleton'):
                skeleton = {
                    'label': skeleton_elem.get('label', 'person'),
                    'elements': []
                }

                for points_elem in skeleton_elem.findall('points'):
                    points_str = points_elem.get('points', '')
                    if ',' in points_str:
                        x, y = map(float, points_str.split(','))
                        skeleton['elements'].append({
                            'label': points_elem.get('label', 'unknown'),
                            'points': [x, y],
                            'outside': points_elem.get('outside', '0') == '1',
                            'occluded': points_elem.get('occluded', '0') == '1'
                        })

                if skeleton['elements']:
                    annotations[image_name]['skeletons'].append(skeleton)

            # Load mask annotations
            for mask_elem in image_elem.findall('mask'):
                rle_str = mask_elem.get('rle', '')
                rle = [int(x) for x in rle_str.split(',')] if rle_str else []
                annotations[image_name]['masks'].append({
                    'label': mask_elem.get('label', 'hand'),
                    'rle': rle,
                    'left': int(mask_elem.get('left', 0)),
                    'top': int(mask_elem.get('top', 0)),
                    'width': int(mask_elem.get('width', 0)),
                    'height': int(mask_elem.get('height', 0))
                })

        return annotations

    def _load_semantic_masks(self, masks_path: Path) -> Dict:
        """Load semantic segmentation masks from JSON file (SAM/Detectron2 results)."""
        with open(masks_path, 'r') as f:
            data = json.load(f)

        masks = {}
        # Handle different result formats
        if isinstance(data, list):
            for item in data:
                image_name = item.get('image', '')
                if 'masks' in item:
                    masks[image_name] = item['masks']
        elif isinstance(data, dict):
            # Assume format: {image_name: {masks: [...]}}
            for image_name, item in data.items():
                if isinstance(item, dict) and 'masks' in item:
                    masks[image_name] = item['masks']
                elif isinstance(item, list):
                    masks[image_name] = item

        return masks

    def load_images(self) -> List[Path]:
        """Load all images from dataset directory."""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        images = [
            f for f in self.dataset_path.iterdir()
            if f.is_file() and f.suffix.lower() in image_extensions
        ]
        return sorted(images)

    def call_mediapipe(self, image_bytes: bytes) -> Optional[List[Dict]]:
        """Call MediaPipe service with image."""
        try:
            image_b64 = base64.b64encode(image_bytes).decode('utf-8')
            payload = {'image': image_b64, 'threshold': 0.3}

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

    def extract_skeleton_keypoints(self, annotation: Dict) -> Dict[str, Dict]:
        """Extract keypoints from skeleton annotation."""
        keypoints = {}
        if annotation.get('type') == 'skeleton' or 'elements' in annotation:
            elements = annotation.get('elements', [])
            for element in elements:
                if element.get('type') == 'points' or 'points' in element:
                    points = element.get('points', [])
                    if len(points) >= 2:
                        label = element.get('label', 'unknown')
                        keypoints[label] = {
                            'x': points[0],
                            'y': points[1],
                            'outside': element.get('outside', False),
                            'occluded': element.get('occluded', False),
                            'confidence': self._get_confidence(element)
                        }
        return keypoints

    def extract_semantic_masks_from_gt(self, annotation_data: Dict, image_width: int, image_height: int) -> List[np.ndarray]:
        """Extract semantic segmentation masks from ground truth annotations."""
        masks = []

        if 'masks' in annotation_data:
            for mask_data in annotation_data['masks']:
                if 'rle' in mask_data:
                    # Decode RLE mask
                    rle = mask_data['rle']
                    mask = self._decode_rle(rle, image_width, image_height)
                    masks.append(mask)
                elif 'polygon' in mask_data:
                    # Convert polygon to mask
                    points = mask_data['polygon']
                    mask = self._polygon_to_mask(points, image_width, image_height)
                    masks.append(mask)

        return masks

    def extract_semantic_masks_from_results(self, image_name: str, image_width: int, image_height: int) -> List[np.ndarray]:
        """Extract semantic segmentation masks from SAM/Detectron2 results."""
        masks = []

        if image_name in self.semantic_masks:
            mask_data_list = self.semantic_masks[image_name]
            for mask_data in mask_data_list:
                if isinstance(mask_data, dict):
                    # Handle different mask formats
                    if 'rle' in mask_data:
                        rle = mask_data['rle']
                        mask = self._decode_rle(rle, image_width, image_height)
                        masks.append(mask)
                    elif 'polygon' in mask_data:
                        points = mask_data['polygon']
                        mask = self._polygon_to_mask(points, image_width, image_height)
                        masks.append(mask)
                    elif 'mask' in mask_data:
                        # Direct mask array (base64 encoded or numpy array)
                        mask_array = mask_data['mask']
                        if isinstance(mask_array, str):
                            # Base64 encoded
                            mask_bytes = base64.b64decode(mask_array)
                            mask = np.frombuffer(mask_bytes, dtype=np.uint8).reshape((image_height, image_width))
                        else:
                            mask = np.array(mask_array, dtype=np.uint8)
                        masks.append(mask)
                elif isinstance(mask_data, list):
                    # Assume RLE format
                    mask = self._decode_rle(mask_data, image_width, image_height)
                    masks.append(mask)

        return masks

    def _decode_rle(self, rle: List[int], width: int, height: int) -> np.ndarray:
        """Decode RLE (Run-Length Encoding) to binary mask."""
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

    def calculate_keypoint_distance(self, gt_kp: Dict, mp_kp: Dict,
                                   image_width: int, image_height: int) -> Tuple[float, float]:
        """Calculate distance between ground truth and MediaPipe keypoint."""
        gt_x, gt_y = gt_kp['x'], gt_kp['y']
        mp_x, mp_y = mp_kp['x'], mp_kp['y']

        dx = abs(gt_x - mp_x)
        dy = abs(gt_y - mp_y)
        distance = np.sqrt(dx**2 + dy**2)

        # Normalize by image size
        max_dim = max(image_width, image_height)
        normalized_distance = distance / max_dim if max_dim > 0 else 0

        return distance, normalized_distance

    def calculate_hand_mask_intersection(self, hand_keypoints: Dict[str, Dict],
                                        semantic_masks: List[np.ndarray],
                                        image_width: int, image_height: int) -> Dict:
        """Calculate intersection of hand keypoints with semantic segmentation masks."""
        if not hand_keypoints or not semantic_masks:
            return {'iou': 0.0, 'intersection_area': 0, 'union_area': 0}

        # Create hand bounding box from keypoints
        hand_points = [(kp['x'], kp['y']) for kp in hand_keypoints.values()
                      if not kp.get('outside', False)]

        if not hand_points:
            return {'iou': 0.0, 'intersection_area': 0, 'union_area': 0}

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

        # Calculate IoU with each semantic mask
        best_iou = 0.0
        best_intersection = 0
        best_union = 0

        for sem_mask in semantic_masks:
            # Ensure masks are same size
            if sem_mask.shape != hand_mask.shape:
                sem_mask_resized = cv2.resize(sem_mask.astype(np.uint8),
                                             (image_width, image_height),
                                             interpolation=cv2.INTER_NEAREST)
            else:
                sem_mask_resized = sem_mask.astype(np.uint8)

            intersection = np.logical_and(hand_mask, sem_mask_resized).sum()
            union = np.logical_or(hand_mask, sem_mask_resized).sum()
            iou = intersection / union if union > 0 else 0.0

            if iou > best_iou:
                best_iou = iou
                best_intersection = intersection
                best_union = union

        return {
            'iou': best_iou,
            'intersection_area': int(best_intersection),
            'union_area': int(best_union)
        }

    def compare_with_ground_truth(self, image_name: str, gt_anno_data: Dict,
                                 mp_result: List[Dict], image_width: int, image_height: int) -> Dict:
        """Compare MediaPipe results with ground truth annotations."""
        comparison = {
            'image': image_name,
            'gt_keypoints': {},
            'mp_keypoints': {},
            'matches': [],
            'missing_in_mp': [],
            'extra_in_mp': [],
            'keypoint_errors': [],
            'hand_detection': {'gt': False, 'mp': False},
            'intersection_metrics': {}
        }

        # Extract ground truth skeleton keypoints
        gt_skeletons = gt_anno_data.get('skeletons', [])
        for skeleton in gt_skeletons:
            kps = self.extract_skeleton_keypoints(skeleton)
            comparison['gt_keypoints'].update(kps)

            # Check if hands are present in GT
            hand_labels = [l for l in kps.keys() if 'hand' in l.lower() or
                          'wrist' in l.lower() or 'thumb' in l.lower() or
                          'finger' in l.lower()]
            if hand_labels:
                comparison['hand_detection']['gt'] = True

        # Extract MediaPipe keypoints
        for skeleton in mp_result:
            if skeleton.get('type') == 'skeleton':
                elements = skeleton.get('elements', [])
                for element in elements:
                    if element.get('type') == 'points':
                        points = element.get('points', [])
                        if len(points) >= 2:
                            label = element.get('label', 'unknown')
                            comparison['mp_keypoints'][label] = {
                                'x': points[0],
                                'y': points[1],
                                'outside': element.get('outside', False),
                                'confidence': self._get_confidence(element)
                            }

        # Check MediaPipe hand detection
        mp_hand_labels = [l for l in comparison['mp_keypoints'].keys()
                         if 'hand' in l.lower() or 'wrist' in l.lower() or
                         'thumb' in l.lower() or 'finger' in l.lower()]
        if mp_hand_labels:
            comparison['hand_detection']['mp'] = True

        # Match keypoints by label
        all_labels = set(comparison['gt_keypoints'].keys()) | set(comparison['mp_keypoints'].keys())

        for label in all_labels:
            gt_kp = comparison['gt_keypoints'].get(label)
            mp_kp = comparison['mp_keypoints'].get(label)

            if gt_kp and mp_kp:
                if not gt_kp.get('outside', False):
                    distance, normalized_distance = self.calculate_keypoint_distance(
                        gt_kp, mp_kp, image_width, image_height
                    )

                    comparison['matches'].append({
                        'label': label,
                        'distance': distance,
                        'normalized_distance': normalized_distance,
                        'gt_pos': (gt_kp['x'], gt_kp['y']),
                        'mp_pos': (mp_kp['x'], mp_kp['y'])
                    })

                    comparison['keypoint_errors'].append({
                        'label': label,
                        'distance': distance,
                        'normalized_distance': normalized_distance
                    })
            elif gt_kp and not mp_kp:
                comparison['missing_in_mp'].append({'label': label, 'gt': gt_kp})
            elif mp_kp and not gt_kp:
                comparison['extra_in_mp'].append({'label': label, 'mp': mp_kp})

        # Calculate intersection with semantic segmentation
        semantic_masks = []

        # Get masks from ground truth annotations
        gt_masks = self.extract_semantic_masks_from_gt(gt_anno_data, image_width, image_height)
        semantic_masks.extend(gt_masks)

        # Get masks from SAM/Detectron2 results
        result_masks = self.extract_semantic_masks_from_results(image_name, image_width, image_height)
        semantic_masks.extend(result_masks)

        if semantic_masks and comparison['mp_keypoints']:
            # Filter hand keypoints
            hand_kps = {k: v for k, v in comparison['mp_keypoints'].items()
                       if 'hand' in k.lower() or 'wrist' in k.lower() or
                       'thumb' in k.lower() or 'finger' in k.lower()}

            if hand_kps:
                intersection = self.calculate_hand_mask_intersection(
                    hand_kps, semantic_masks, image_width, image_height
                )
                comparison['intersection_metrics'] = intersection

        return comparison

    def analyze_dataset(self):
        """Analyze entire dataset."""
        logger.info(f"Analyzing dataset: {self.dataset_path}")

        images = self.load_images()
        if not images:
            logger.error(f"No images found in {self.dataset_path}")
            return

        logger.info(f"Found {len(images)} images")

        results = []

        for image_path in images:
            image_name = image_path.name
            logger.info(f"Processing {image_name}...")

            # Load image
            with open(image_path, 'rb') as f:
                image_bytes = f.read()

            image = Image.open(io.BytesIO(image_bytes))
            image_width, image_height = image.size

            # Get ground truth annotations
            gt_anno_data = self.gt_annotations.get(image_name, {})
            has_gt = bool(gt_anno_data.get('skeletons') or gt_anno_data.get('masks'))

            # Call MediaPipe
            mp_result = self.call_mediapipe(image_bytes)
            has_mp = mp_result and len(mp_result) > 0

            # Update statistics
            self.stats['total_images'] += 1
            if has_gt:
                self.stats['images_with_gt'] += 1
            if has_mp:
                self.stats['images_with_mp'] += 1
            if has_gt and has_mp:
                self.stats['images_with_both'] += 1

            # Compare if both exist
            if has_gt and has_mp:
                comparison = self.compare_with_ground_truth(
                    image_name, gt_anno_data, mp_result, image_width, image_height
                )
                results.append(comparison)

                # Update statistics
                if comparison['hand_detection']['gt']:
                    if comparison['hand_detection']['mp']:
                        self.stats['hand_detection_rate'] += 1

                self.stats['keypoint_accuracy'].extend(comparison['keypoint_errors'])
                self.stats['keypoint_distance_errors'].extend(
                    [e['normalized_distance'] for e in comparison['keypoint_errors']]
                )

                if comparison['intersection_metrics']:
                    self.stats['intersection_metrics'].append(comparison['intersection_metrics'])

        # Calculate final statistics
        if self.stats['images_with_gt'] > 0:
            self.stats['hand_detection_rate'] /= self.stats['images_with_gt']

        # Save results
        results_path = self.output_dir / f"mediapipe_ground_truth_test_{int(time.time())}.json"
        with open(results_path, 'w') as f:
            json.dump({
                'statistics': self.stats,
                'detailed_results': results,
                'dataset_path': str(self.dataset_path),
                'annotations_path': str(self.annotations_path) if self.annotations_path else None,
                'semantic_masks_path': str(self.semantic_masks_path) if self.semantic_masks_path else None
            }, f, indent=2)

        logger.info(f"Analysis complete. Results saved to {results_path}")
        self.print_summary()

    def print_summary(self):
        """Print analysis summary."""
        print("\n" + "="*80)
        print("GROUND TRUTH VALIDATION SUMMARY")
        print("="*80)
        print(f"Total images analyzed: {self.stats['total_images']}")
        print(f"Images with ground truth: {self.stats['images_with_gt']}")
        print(f"Images with MediaPipe detections: {self.stats['images_with_mp']}")
        print(f"Images with both: {self.stats['images_with_both']}")
        print(f"Hand detection rate: {self.stats['hand_detection_rate']:.2%}")

        if self.stats['keypoint_distance_errors']:
            avg_error = np.mean(self.stats['keypoint_distance_errors'])
            median_error = np.median(self.stats['keypoint_distance_errors'])
            p95_error = np.percentile(self.stats['keypoint_distance_errors'], 95)
            print(f"\nKeypoint Accuracy:")
            print(f"  Average error: {avg_error:.2%} of image size")
            print(f"  Median error: {median_error:.2%} of image size")
            print(f"  95th percentile error: {p95_error:.2%} of image size")

        if self.stats['intersection_metrics']:
            avg_iou = np.mean([m['iou'] for m in self.stats['intersection_metrics']])
            print(f"\nSemantic Segmentation Intersection:")
            print(f"  Average IoU: {avg_iou:.3f}")

        print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Test MediaPipe against ground truth annotations from local files"
    )
    parser.add_argument('--dataset-path', required=True,
                       help='Path to dataset directory with images')
    parser.add_argument('--annotations', help='Path to ground truth annotations file (JSON or XML)')
    parser.add_argument('--semantic-masks', help='Path to semantic segmentation masks (SAM/Detectron2 results JSON)')
    parser.add_argument('--mediapipe-url', default='http://localhost:8000',
                       help='MediaPipe service URL')
    parser.add_argument('--output-dir', default='test-results',
                       help='Output directory for results')

    args = parser.parse_args()

    tester = MediaPipeGroundTruthTester(
        dataset_path=args.dataset_path,
        annotations_path=args.annotations,
        semantic_masks_path=args.semantic_masks,
        mediapipe_url=args.mediapipe_url,
        output_dir=args.output_dir
    )

    tester.analyze_dataset()


if __name__ == '__main__':
    main()
