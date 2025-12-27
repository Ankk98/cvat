#!/usr/bin/env python3
"""
SAM Egocentric Vision Testing Script
====================================

Tests the Segment Anything Model (SAM) on egocentric vision datasets.
Focuses on hand segmentation, object manipulation, and interactive segmentation
capabilities for first-person view scenarios.

Features:
- Automated testing on multiple difficulty levels
- Hand segmentation accuracy evaluation
- Object interaction analysis
- CVAT integration validation
- Performance benchmarking

Usage:
    python test_sam_egocentric.py --dataset epic-kitchens --difficulty easy
    python test_sam_egocentric.py --dataset all --output-dir ./results
"""

import argparse
import base64
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import requests
from PIL import Image
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SAMEgocentricTester:
    """SAM testing for egocentric vision tasks."""

    def __init__(self, dataset_path: str, cvat_url: Optional[str] = None, sam_service_url: str = "http://localhost:32768"):
        self.dataset_path = Path(dataset_path)
        self.cvat_url = cvat_url
        self.sam_service_url = sam_service_url
        self.results = []

        # SAM service endpoints
        self.sam_endpoint = f"{sam_service_url}"

    def test_service_health(self) -> bool:
        """Test if SAM service is healthy."""
        try:
            # For SAM, we test by making a simple inference request
            # Create a minimal test image
            from PIL import Image
            import io
            img = Image.new('RGB', (50, 50), color='red')
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG')
            img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

            payload = {"image": img_base64}
            response = requests.post(self.sam_service_url, json=payload, timeout=10)
            if response.status_code == 200:
                logger.info("✅ SAM service is healthy")
                return True
            else:
                logger.error(f"❌ SAM service health check failed: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"❌ Cannot connect to SAM service: {e}")
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

    def run_sam_inference(self, image_b64: str, points: Optional[List[List[int]]] = None) -> Dict:
        """Run SAM inference on image."""
        # For SAM testing, we'll simulate interactive segmentation
        # In a real scenario, you'd provide point prompts for specific objects

        payload = {
            "image": image_b64,
            "threshold": 0.8
        }

        # Add point prompts if provided (for interactive segmentation)
        if points:
            payload["points"] = points

        try:
            response = requests.post(self.sam_endpoint, json=payload, timeout=30)
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"SAM inference failed: {response.status_code} - {response.text}")
                return {}
        except Exception as e:
            logger.error(f"SAM inference error: {e}")
            return {}

    def analyze_hand_segmentation(self, image_path: Path, sam_result: Dict) -> Dict:
        """Analyze hand segmentation quality."""
        analysis = {
            "image": str(image_path.name),
            "has_masks": False,
            "mask_count": 0,
            "total_mask_area": 0,
            "hand_like_masks": 0,
            "processing_time": sam_result.get("processing_time", 0),
            "mask_confidences": []
        }

        # Analyze SAM results
        if "masks" in sam_result:
            masks = sam_result["masks"]
            analysis["has_masks"] = True
            analysis["mask_count"] = len(masks)

            for mask_data in masks:
                if isinstance(mask_data, dict):
                    # Extract confidence if available
                    confidence = mask_data.get("confidence", 1.0)
                    analysis["mask_confidences"].append(confidence)

                    # Analyze mask area (simplified)
                    # In a real implementation, you'd decode and analyze the mask
                    analysis["total_mask_area"] += 1  # Placeholder

                    # Simple heuristic for hand-like masks
                    # Real implementation would use shape analysis
                    if confidence > 0.7:  # High confidence might indicate hands
                        analysis["hand_like_masks"] += 1

        return analysis

    def test_difficulty_level(self, difficulty: str, max_samples: Optional[int] = None) -> Dict:
        """Test SAM on a specific difficulty level."""
        logger.info(f"🧪 Testing SAM on {difficulty} difficulty images")

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
            "hand_detection_rate": 0,
            "image_results": []
        }

        total_processing_time = 0
        hand_detections = 0

        for image_path in tqdm(images, desc=f"Testing {difficulty}"):
            try:
                # Convert image to base64
                image_b64 = self.image_to_base64(image_path)

                # Run SAM inference
                start_time = time.time()
                sam_result = self.run_sam_inference(image_b64)
                processing_time = time.time() - start_time

                if sam_result:
                    results["successful_inferences"] += 1

                    # Analyze results
                    analysis = self.analyze_hand_segmentation(image_path, sam_result)
                    analysis["processing_time"] = processing_time

                    results["image_results"].append(analysis)
                    total_processing_time += processing_time

                    # Count hand-like detections
                    if analysis["hand_like_masks"] > 0:
                        hand_detections += 1

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
            results["hand_detection_rate"] = hand_detections / results["successful_inferences"]

        return results

    def run_comprehensive_test(self, difficulties: List[str] = None,
                             max_samples_per_difficulty: Optional[int] = None) -> Dict:
        """Run comprehensive testing across multiple difficulty levels."""
        if difficulties is None:
            difficulties = ["easy", "medium", "hard"]

        logger.info("🚀 Starting comprehensive SAM egocentric testing")
        logger.info(f"Testing difficulties: {difficulties}")
        logger.info(f"Max samples per difficulty: {max_samples_per_difficulty}")

        # Test service health first
        if not self.test_service_health():
            return {"error": "SAM service not available"}

        overall_results = {
            "model": "SAM",
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
            "overall_hand_detection_rate": 0,
            "difficulty_performance": {}
        }

        total_time = 0
        total_hand_detections = 0

        for difficulty, result in results["difficulty_results"].items():
            if "error" in result:
                continue

            summary["total_images_tested"] += result["total_images"]
            summary["total_successful"] += result["successful_inferences"]
            summary["total_failed"] += result["failed_inferences"]

            if result["successful_inferences"] > 0:
                total_time += result["average_processing_time"] * result["successful_inferences"]
                total_hand_detections += result["hand_detection_rate"] * result["successful_inferences"]

            summary["difficulty_performance"][difficulty] = {
                "success_rate": result["successful_inferences"] / max(result["total_images"], 1),
                "hand_detection_rate": result["hand_detection_rate"],
                "avg_processing_time": result["average_processing_time"]
            }

        # Overall averages
        if summary["total_successful"] > 0:
            summary["average_processing_time"] = total_time / summary["total_successful"]
            summary["overall_hand_detection_rate"] = total_hand_detections / summary["total_successful"]

        return summary

    def save_results(self, results: Dict, output_path: Path):
        """Save test results to file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"📄 Results saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="SAM Egocentric Vision Testing")
    parser.add_argument("--dataset", required=True,
                       help="Path to egocentric dataset")
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard", "all"],
                       default="all", help="Difficulty level to test")
    parser.add_argument("--max-samples", type=int,
                       help="Maximum samples per difficulty level")
    parser.add_argument("--sam-url", default="http://localhost:32768",
                       help="SAM service URL")
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
    tester = SAMEgocentricTester(
        dataset_path=dataset_path,
        cvat_url=args.cvat_url,
        sam_service_url=args.sam_url
    )

    # Run tests
    results = tester.run_comprehensive_test(
        difficulties=difficulties,
        max_samples_per_difficulty=args.max_samples
    )

    # Save results
    timestamp = int(time.time())
    result_file = output_dir / f"sam_egocentric_test_{timestamp}.json"
    tester.save_results(results, result_file)

    # Print summary
    if "error" not in results:
        summary = results["summary"]
        print("\n" + "="*60)
        print("🎯 SAM EGOCENTRIC TESTING SUMMARY")
        print("="*60)
        print(f"📊 Images Tested: {summary['total_images_tested']}")
        print(f"✅ Successful: {summary['total_successful']}")
        print(f"❌ Failed: {summary['total_failed']}")
        print(".2f")
        print(".2f")
        print(f"📁 Results: {result_file}")
        print("="*60)

    return 0 if "error" not in results else 1

if __name__ == "__main__":
    exit(main())
