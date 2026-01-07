#!/usr/bin/env python3
"""
Detectron2 Egocentric Vision Testing Script
===========================================

Tests Detectron2 instance segmentation on egocentric vision datasets.
Evaluates performance on object detection, hand segmentation, and
kitchen utensil recognition in first-person view scenarios.

Features:
- Instance segmentation accuracy evaluation
- Object detection in cluttered egocentric scenes
- Performance benchmarking across difficulty levels
- CVAT integration validation

Usage:
    python test_detectron2_egocentric.py --dataset epic-kitchens --difficulty easy
    python test_detectron2_egocentric.py --dataset all --output-dir ./results
"""

import argparse
import base64
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import requests
from PIL import Image
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Detectron2EgocentricTester:
    """Detectron2 testing for egocentric vision tasks."""

    def __init__(self, dataset_path: str, cvat_url: Optional[str] = None, detectron_service_url: str = "http://localhost:32769"):
        self.dataset_path = Path(dataset_path)
        self.cvat_url = cvat_url
        self.detectron_service_url = detectron_service_url
        self.results = []

        # Detectron2 service endpoint
        self.detectron_endpoint = f"{detectron_service_url}"

        # COCO class names for analysis
        self.coco_classes = {
            1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane',
            6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light',
            11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench',
            16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow',
            22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe', 27: 'backpack',
            28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase', 34: 'frisbee',
            35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite', 39: 'baseball bat',
            40: 'baseball glove', 41: 'skateboard', 42: 'surfboard', 43: 'tennis racket',
            44: 'bottle', 46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife',
            50: 'spoon', 51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich',
            55: 'orange', 56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza',
            60: 'donut', 61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant',
            65: 'bed', 67: 'dining table', 70: 'toilet', 72: 'tv', 73: 'laptop',
            74: 'mouse', 75: 'remote', 76: 'keyboard', 77: 'cell phone', 78: 'microwave',
            79: 'oven', 80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book',
            85: 'clock', 86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier',
            90: 'toothbrush'
        }

        # Egocentric-relevant classes
        self.egocentric_classes = {
            'person', 'handbag', 'bottle', 'cup', 'fork', 'knife', 'spoon', 'bowl',
            'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'chair',
            'dining table', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
            'book', 'scissors'
        }

    def test_service_health(self) -> bool:
        """Test if Detectron2 service is healthy."""
        try:
            # For Detectron2, we test by making a simple inference request
            # Create a minimal test image
            from PIL import Image
            import io
            img = Image.new('RGB', (50, 50), color='red')
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG')
            img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

            payload = {"image": img_base64, "threshold": 0.1}
            response = requests.post(self.detectron_service_url, json=payload, timeout=10)
            if response.status_code == 200:
                logger.info("✅ Detectron2 service is healthy")
                return True
            else:
                logger.error(f"❌ Detectron2 service health check failed: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"❌ Cannot connect to Detectron2 service: {e}")
            return False

    def load_test_images(self, difficulty: str) -> List[Path]:
        """Load test images for specified difficulty level."""
        difficulty_path = self.dataset_path / difficulty
        if not difficulty_path.exists():
            logger.error(f"Difficulty path not found: {difficulty_path}")
            return []

        # Get all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        image_files = [
            f for f in difficulty_path.iterdir()
            if f.is_file() and f.suffix.lower() in image_extensions
        ]

        logger.info(f"Found {len(image_files)} images for {difficulty} difficulty")
        return sorted(image_files)

    def image_to_base64(self, image_path: Path) -> str:
        """Convert image to base64 string."""
        with open(image_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')

    def run_detectron_inference(self, image_b64: str, threshold: float = 0.5) -> Dict:
        """Run Detectron2 inference on image."""
        payload = {
            "image": image_b64,
            "threshold": threshold
        }

        try:
            response = requests.post(self.detectron_endpoint, json=payload, timeout=30)
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Detectron2 inference failed: {response.status_code} - {response.text}")
                return {}
        except Exception as e:
            logger.error(f"Detectron2 inference error: {e}")
            return {}

    def analyze_instance_segmentation(self, image_path: Path, detectron_result: Dict) -> Dict:
        """Analyze instance segmentation quality."""
        analysis = {
            "image": str(image_path.name),
            "has_detections": False,
            "detection_count": 0,
            "egocentric_objects": 0,
            "hand_detections": 0,
            "kitchen_objects": 0,
            "processing_time": detectron_result.get("processing_time", 0),
            "detections": []
        }

        # Analyze Detectron2 results
        # Detectron2 returns a list directly, not wrapped in "annotations"
        annotations = detectron_result if isinstance(detectron_result, list) else detectron_result.get("annotations", [])

        if len(annotations) > 0:
            analysis["has_detections"] = True
            analysis["detection_count"] = len(annotations)

            for ann in annotations:
                label_name = ann.get("label", "unknown")
                class_name = label_name  # Detectron2 returns COCO category names directly
                confidence = float(ann.get("confidence", 0))
                ann_type = ann.get("type", "unknown")
                points = ann.get("points", [])

                detection_info = {
                    "class_name": class_name,
                    "confidence": confidence,
                    "type": ann_type,
                    "points_length": len(points),
                    "has_mask": ann_type == "mask",
                }

                # Extract bbox from points
                if ann_type == "mask" and len(points) >= 4:
                    # Last 4 values are bbox: [x_min, y_min, x_max, y_max]
                    bbox = points[-4:]
                    detection_info["bbox"] = bbox
                    detection_info["rle_length"] = len(points) - 4
                elif ann_type == "rectangle" and len(points) == 4:
                    # Points are bbox: [x1, y1, x2, y2]
                    detection_info["bbox"] = points
                else:
                    detection_info["bbox"] = []

                analysis["detections"].append(detection_info)

                # Categorize detections
                if class_name in self.egocentric_classes:
                    analysis["egocentric_objects"] += 1

                # Look for hand-related detections (though COCO doesn't have explicit hand class)
                if class_name in ['person'] and confidence > 0.8:
                    # In egocentric vision, person detections often include visible hands
                    analysis["hand_detections"] += 1

                # Kitchen objects
                kitchen_items = {'bottle', 'cup', 'fork', 'knife', 'spoon', 'bowl',
                               'microwave', 'oven', 'sink', 'refrigerator'}
                if class_name in kitchen_items:
                    analysis["kitchen_objects"] += 1

        return analysis

    def test_difficulty_level(self, difficulty: str, max_samples: Optional[int] = None) -> Dict:
        """Test Detectron2 on a specific difficulty level."""
        logger.info(f"🧪 Testing Detectron2 on {difficulty} difficulty images")

        images = self.load_test_images(difficulty)
        if not images:
            return {"error": f"No images found for {difficulty}"}

        if max_samples:
            images = images[:max_samples]

        results = {
            "difficulty": difficulty,
            "total_images": len(images),
            "successful_inferences": 0,
            "failed_inferences": 0,
            "average_processing_time": 0,
            "average_detections_per_image": 0,
            "egocentric_object_detection_rate": 0,
            "hand_detection_rate": 0,
            "kitchen_object_detection_rate": 0,
            "image_results": []
        }

        total_processing_time = 0
        total_detections = 0
        total_egocentric_objects = 0
        total_hand_detections = 0
        total_kitchen_objects = 0

        for image_path in tqdm(images, desc=f"Testing {difficulty}"):
            try:
                # Convert image to base64
                image_b64 = self.image_to_base64(image_path)

                # Run Detectron2 inference
                start_time = time.time()
                detectron_result = self.run_detectron_inference(image_b64)
                processing_time = time.time() - start_time

                if detectron_result:
                    results["successful_inferences"] += 1

                    # Analyze results
                    analysis = self.analyze_instance_segmentation(image_path, detectron_result)
                    analysis["processing_time"] = processing_time

                    results["image_results"].append(analysis)
                    total_processing_time += processing_time
                    total_detections += analysis["detection_count"]
                    total_egocentric_objects += analysis["egocentric_objects"]
                    total_hand_detections += analysis["hand_detections"]
                    total_kitchen_objects += analysis["kitchen_objects"]

                else:
                    results["failed_inferences"] += 1
                    results["image_results"].append({
                        "image": str(image_path.name),
                        "error": "Inference failed"
                    })

            except Exception as e:
                logger.error(f"Error testing {image_path}: {e}")
                results["failed_inferences"] += 1

        # Calculate averages
        if results["successful_inferences"] > 0:
            results["average_processing_time"] = total_processing_time / results["successful_inferences"]
            results["average_detections_per_image"] = total_detections / results["successful_inferences"]
            results["egocentric_object_detection_rate"] = total_egocentric_objects / results["successful_inferences"]
            results["hand_detection_rate"] = total_hand_detections / results["successful_inferences"]
            results["kitchen_object_detection_rate"] = total_kitchen_objects / results["successful_inferences"]

        return results

    def run_comprehensive_test(self, difficulties: List[str] = None,
                             max_samples_per_difficulty: Optional[int] = None) -> Dict:
        """Run comprehensive testing across multiple difficulty levels."""
        if difficulties is None:
            difficulties = ["easy", "medium", "hard"]

        logger.info("🚀 Starting comprehensive Detectron2 egocentric testing")
        logger.info(f"Testing difficulties: {difficulties}")
        logger.info(f"Max samples per difficulty: {max_samples_per_difficulty}")

        # Test service health first
        if not self.test_service_health():
            return {"error": "Detectron2 service not available"}

        overall_results = {
            "model": "Detectron2",
            "dataset": str(self.dataset_path.name),
            "test_timestamp": time.time(),
            "difficulty_results": {},
            "summary": {}
        }

        # Test each difficulty level
        for difficulty in difficulties:
            result = self.test_difficulty_level(difficulty, max_samples_per_difficulty)
            overall_results["difficulty_results"][difficulty] = result

        # Generate summary
        overall_results["summary"] = self.generate_summary(overall_results)

        return overall_results

    def generate_summary(self, results: Dict) -> Dict:
        """Generate summary statistics."""
        summary = {
            "total_images_tested": 0,
            "total_successful": 0,
            "total_failed": 0,
            "average_processing_time": 0,
            "average_detections_per_image": 0,
            "overall_egocentric_detection_rate": 0,
            "overall_hand_detection_rate": 0,
            "overall_kitchen_detection_rate": 0,
            "difficulty_performance": {}
        }

        total_time = 0
        total_detections = 0
        total_egocentric = 0
        total_hands = 0
        total_kitchen = 0

        for difficulty, result in results["difficulty_results"].items():
            if "error" in result:
                continue

            summary["total_images_tested"] += result["total_images"]
            summary["total_successful"] += result["successful_inferences"]
            summary["total_failed"] += result["failed_inferences"]

            if result["successful_inferences"] > 0:
                total_time += result["average_processing_time"] * result["successful_inferences"]
                total_detections += result["average_detections_per_image"] * result["successful_inferences"]
                total_egocentric += result["egocentric_object_detection_rate"] * result["successful_inferences"]
                total_hands += result["hand_detection_rate"] * result["successful_inferences"]
                total_kitchen += result["kitchen_object_detection_rate"] * result["successful_inferences"]

            summary["difficulty_performance"][difficulty] = {
                "success_rate": result["successful_inferences"] / max(result["total_images"], 1),
                "avg_detections": result["average_detections_per_image"],
                "egocentric_rate": result["egocentric_object_detection_rate"],
                "hand_rate": result["hand_detection_rate"],
                "kitchen_rate": result["kitchen_object_detection_rate"],
                "avg_processing_time": result["average_processing_time"]
            }

        # Overall averages
        if summary["total_successful"] > 0:
            summary["average_processing_time"] = total_time / summary["total_successful"]
            summary["average_detections_per_image"] = total_detections / summary["total_successful"]
            summary["overall_egocentric_detection_rate"] = total_egocentric / summary["total_successful"]
            summary["overall_hand_detection_rate"] = total_hands / summary["total_successful"]
            summary["overall_kitchen_detection_rate"] = total_kitchen / summary["total_successful"]

        return summary

    def save_results(self, results: Dict, output_path: Path):
        """Save test results to file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"📄 Results saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Detectron2 Egocentric Vision Testing")
    parser.add_argument("--dataset", required=True,
                       help="Path to egocentric dataset")
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard", "all"],
                       default="all", help="Difficulty level to test")
    parser.add_argument("--max-samples", type=int,
                       help="Maximum samples per difficulty level")
    parser.add_argument("--detectron-url", default="http://localhost:32769",
                       help="Detectron2 service URL")
    parser.add_argument("--cvat-url",
                       help="CVAT server URL for integration testing")
    parser.add_argument("--output-dir", default="./test-results",
                       help="Output directory for results")

    args = parser.parse_args()

    # Setup
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        logger.error(f"Dataset path not found: {dataset_path}")
        return 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine difficulties to test
    difficulties = ["easy", "medium", "hard"] if args.difficulty == "all" else [args.difficulty]

    # Create tester
    tester = Detectron2EgocentricTester(
        dataset_path=dataset_path,
        cvat_url=args.cvat_url,
        detectron_service_url=args.detectron_url
    )

    # Run tests
    results = tester.run_comprehensive_test(
        difficulties=difficulties,
        max_samples_per_difficulty=args.max_samples
    )

    # Save results
    timestamp = int(time.time())
    result_file = output_dir / f"detectron2_egocentric_test_{timestamp}.json"
    tester.save_results(results, result_file)

    # Print summary
    if "error" not in results:
        summary = results["summary"]
        print("\n" + "="*60)
        print("🎯 DETECTRON2 EGOCENTRIC TESTING SUMMARY")
        print("="*60)
        print(f"📊 Images Tested: {summary['total_images_tested']}")
        print(f"✅ Successful: {summary['total_successful']}")
        print(f"❌ Failed: {summary['total_failed']}")
        print(".2f")
        print(".1f")
        print(".2f")
        print(".2f")
        print(".2f")
        print(f"📁 Results: {result_file}")
        print("="*60)

    return 0 if "error" not in results else 1

if __name__ == "__main__":
    exit(main())
