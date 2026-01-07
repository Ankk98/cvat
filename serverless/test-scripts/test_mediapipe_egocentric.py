#!/usr/bin/env python3
"""
MediaPipe Pose+Hands Egocentric Vision Testing Script
=====================================================

Tests MediaPipe comprehensive pose and hand detection on egocentric vision datasets.
Evaluates full-body pose estimation combined with detailed finger joint tracking
for first-person view scenarios.

Features:
- 33-point pose estimation accuracy
- 42-point hand/finger joint detection
- Combined pose+hands performance analysis
- Egocentric vision specific metrics
- CVAT integration validation

Usage:
    python test_mediapipe_egocentric.py --dataset epic-kitchens --difficulty easy
    python test_mediapipe_egocentric.py --dataset all --include-hands --output-dir ./results
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

class MediaPipeEgocentricTester:
    """MediaPipe Pose+Hands testing for egocentric vision tasks."""

    def __init__(self, dataset_path: str, cvat_url: Optional[str] = None, mediapipe_service_url: str = "http://localhost:8000"):
        self.dataset_path = Path(dataset_path)
        self.cvat_url = cvat_url
        self.mediapipe_service_url = mediapipe_service_url
        self.results = []

        # MediaPipe service endpoints
        self.mediapipe_endpoint = f"{mediapipe_service_url}/detect"

    def test_service_health(self) -> bool:
        """Test if MediaPipe service is healthy."""
        try:
            response = requests.get(f"{self.mediapipe_service_url}/health", timeout=10)
            if response.status_code == 200:
                logger.info("✅ MediaPipe service is healthy")
                return True
            else:
                logger.error(f"❌ MediaPipe service health check failed: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"❌ Cannot connect to MediaPipe service: {e}")
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

    def run_mediapipe_inference(self, image_b64: str, threshold: float = 0.3) -> Dict:
        """Run MediaPipe pose+hands inference on image."""
        payload = {
            "image": image_b64,
            "threshold": threshold
        }

        try:
            response = requests.post(self.mediapipe_endpoint, json=payload, timeout=30)
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"MediaPipe inference failed: {response.status_code} - {response.text}")
                return {}
        except Exception as e:
            logger.error(f"MediaPipe inference error: {e}")
            return {}

    def analyze_pose_hands_results(self, image_path: Path, mediapipe_result: Dict) -> Dict:
        """Analyze MediaPipe pose+hands detection quality."""
        analysis = {
            "image": str(image_path.name),
            "has_pose": False,
            "has_hands": False,
            "pose_keypoints": 0,
            "hand_keypoints": 0,
            "total_keypoints": 0,
            "body_visible": False,
            "hands_visible": False,
            "left_hand_detected": False,
            "right_hand_detected": False,
            "finger_joints_detected": 0,
            "processing_time": mediapipe_result.get("processing_time", 0),
            "keypoint_details": {}
        }

        # Analyze MediaPipe results (should be a list with one skeleton)
        if isinstance(mediapipe_result, list) and len(mediapipe_result) > 0:
            skeleton = mediapipe_result[0]
            elements = skeleton.get("elements", [])

            # Categorize keypoints
            body_keypoints = []
            hand_keypoints = []

            for element in elements:
                label = element.get("label", "")

                # Body keypoints (pose estimation)
                if any(body_part in label for body_part in [
                    "nose", "eye", "ear", "shoulder", "elbow", "wrist", "hip", "knee", "ankle"
                ]):
                    body_keypoints.append(element)

                # Hand keypoints (finger joints and bases)
                elif any(hand_part in label for hand_part in [
                    "wrist", "thumb", "index", "middle", "ring", "pinky", "_cmc", "_mcp", "_pip", "_dip", "_ip", "_tip"
                ]):
                    hand_keypoints.append(element)

            # Update analysis
            analysis["pose_keypoints"] = len(body_keypoints)
            analysis["hand_keypoints"] = len(hand_keypoints)
            analysis["total_keypoints"] = len(elements)
            analysis["has_pose"] = len(body_keypoints) > 10  # At least some body keypoints
            analysis["has_hands"] = len(hand_keypoints) > 0

            # Check body visibility (major joints present)
            major_body_joints = ["nose", "left_shoulder", "right_shoulder", "left_hip", "right_hip"]
            analysis["body_visible"] = any(
                any(joint in elem["label"] for joint in major_body_joints)
                for elem in body_keypoints
            )

            # Check hand visibility and specific hand detection
            left_hand_joints = [kp for kp in hand_keypoints if kp["label"].startswith("left_")]
            right_hand_joints = [kp for kp in hand_keypoints if kp["label"].startswith("right_")]

            analysis["left_hand_detected"] = len(left_hand_joints) > 3
            analysis["right_hand_detected"] = len(right_hand_joints) > 3
            analysis["hands_visible"] = analysis["left_hand_detected"] or analysis["right_hand_detected"]

            # Count individual finger joints (exclude wrist and palm bases)
            finger_joints = [kp for kp in hand_keypoints if any(joint in kp["label"]
                            for joint in ["_mcp", "_pip", "_dip", "_ip", "_tip", "_cmc"])]
            analysis["finger_joints_detected"] = len(finger_joints)

            # Detailed keypoint information
            analysis["keypoint_details"] = {
                "body": {
                    "total": len(body_keypoints),
                    "joints": [kp["label"] for kp in body_keypoints[:5]]  # Sample first 5
                },
                "hands": {
                    "total": len(hand_keypoints),
                    "left_hand": len(left_hand_joints),
                    "right_hand": len(right_hand_joints),
                    "finger_joints": len(finger_joints)
                }
            }

        return analysis

    def test_difficulty_level(self, difficulty: str, max_samples: Optional[int] = None) -> Dict:
        """Test MediaPipe on a specific difficulty level."""
        logger.info(f"🧪 Testing MediaPipe Pose+Hands on {difficulty} difficulty images")

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
            "pose_detection_rate": 0,
            "hand_detection_rate": 0,
            "body_visibility_rate": 0,
            "average_pose_keypoints": 0,
            "average_hand_keypoints": 0,
            "finger_joint_detection_rate": 0,
            "image_results": []
        }

        total_processing_time = 0
        pose_detections = 0
        hand_detections = 0
        body_visibilities = 0
        total_pose_keypoints = 0
        total_hand_keypoints = 0
        finger_joint_detections = 0

        for image_path in tqdm(images, desc=f"Testing {difficulty}"):
            try:
                # Convert image to base64
                image_b64 = self.image_to_base64(image_path)

                # Run MediaPipe inference
                start_time = time.time()
                mediapipe_result = self.run_mediapipe_inference(image_b64)
                processing_time = time.time() - start_time

                if mediapipe_result:
                    results["successful_inferences"] += 1

                    # Analyze results
                    analysis = self.analyze_pose_hands_results(image_path, mediapipe_result)
                    analysis["processing_time"] = processing_time

                    results["image_results"].append(analysis)
                    total_processing_time += processing_time

                    # Update counters
                    if analysis["has_pose"]:
                        pose_detections += 1
                        total_pose_keypoints += analysis["pose_keypoints"]

                    if analysis["hands_visible"]:
                        hand_detections += 1
                        total_hand_keypoints += analysis["hand_keypoints"]

                    if analysis["body_visible"]:
                        body_visibilities += 1

                    if analysis["finger_joints_detected"] > 0:
                        finger_joint_detections += 1

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

        if results["total_images"] > 0:
            results["pose_detection_rate"] = pose_detections / results["total_images"]
            results["hand_detection_rate"] = hand_detections / results["total_images"]
            results["body_visibility_rate"] = body_visibilities / results["total_images"]
            results["finger_joint_detection_rate"] = finger_joint_detections / results["total_images"]

        if pose_detections > 0:
            results["average_pose_keypoints"] = total_pose_keypoints / pose_detections

        if hand_detections > 0:
            results["average_hand_keypoints"] = total_hand_keypoints / hand_detections

        return results

    def run_comprehensive_test(self, difficulties: List[str] = None,
                             max_samples_per_difficulty: Optional[int] = None) -> Dict:
        """Run comprehensive testing across multiple difficulty levels."""
        if difficulties is None:
            difficulties = ["easy", "medium", "hard"]

        logger.info("🚀 Starting comprehensive MediaPipe Pose+Hands egocentric testing")
        logger.info(f"Testing difficulties: {difficulties}")
        logger.info(f"Max samples per difficulty: {max_samples_per_difficulty}")

        # Test service health first
        if not self.test_service_health():
            return {"error": "MediaPipe service not available"}

        overall_results = {
            "model": "MediaPipe_Pose_Hands",
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
            "overall_pose_detection_rate": 0,
            "overall_hand_detection_rate": 0,
            "overall_body_visibility_rate": 0,
            "overall_finger_joint_detection_rate": 0,
            "average_pose_keypoints": 0,
            "average_hand_keypoints": 0,
            "difficulty_performance": {}
        }

        total_time = 0
        total_pose_rate = 0
        total_hand_rate = 0
        total_body_visibility = 0
        total_finger_joints = 0
        total_pose_keypoints = 0
        total_hand_keypoints = 0
        pose_images = 0
        hand_images = 0

        for difficulty, result in results["difficulty_results"].items():
            if "error" in result:
                continue

            summary["total_images_tested"] += result["total_images"]
            summary["total_successful"] += result["successful_inferences"]
            summary["total_failed"] += result["failed_inferences"]

            if result["successful_inferences"] > 0:
                total_time += result["average_processing_time"] * result["successful_inferences"]

            # Accumulate rates weighted by image count
            img_count = result["total_images"]
            if img_count > 0:
                total_pose_rate += result["pose_detection_rate"] * img_count
                total_hand_rate += result["hand_detection_rate"] * img_count
                total_body_visibility += result["body_visibility_rate"] * img_count
                total_finger_joints += result["finger_joint_detection_rate"] * img_count

            # Accumulate keypoints
            if result["pose_detection_rate"] > 0:
                total_pose_keypoints += result["average_pose_keypoints"] * result["pose_detection_rate"] * img_count
                pose_images += result["pose_detection_rate"] * img_count

            if result["hand_detection_rate"] > 0:
                total_hand_keypoints += result["average_hand_keypoints"] * result["hand_detection_rate"] * img_count
                hand_images += result["hand_detection_rate"] * img_count

            summary["difficulty_performance"][difficulty] = {
                "success_rate": result["successful_inferences"] / max(result["total_images"], 1),
                "pose_detection_rate": result["pose_detection_rate"],
                "hand_detection_rate": result["hand_detection_rate"],
                "body_visibility_rate": result["body_visibility_rate"],
                "finger_joint_detection_rate": result["finger_joint_detection_rate"],
                "avg_pose_keypoints": result["average_pose_keypoints"],
                "avg_hand_keypoints": result["average_hand_keypoints"],
                "avg_processing_time": result["average_processing_time"]
            }

        # Overall averages
        total_images = summary["total_images_tested"]
        if total_images > 0:
            summary["overall_pose_detection_rate"] = total_pose_rate / total_images
            summary["overall_hand_detection_rate"] = total_hand_rate / total_images
            summary["overall_body_visibility_rate"] = total_body_visibility / total_images
            summary["overall_finger_joint_detection_rate"] = total_finger_joints / total_images

        if summary["total_successful"] > 0:
            summary["average_processing_time"] = total_time / summary["total_successful"]

        if pose_images > 0:
            summary["average_pose_keypoints"] = total_pose_keypoints / pose_images

        if hand_images > 0:
            summary["average_hand_keypoints"] = total_hand_keypoints / hand_images

        return summary

    def save_results(self, results: Dict, output_path: Path):
        """Save test results to file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"📄 Results saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="MediaPipe Pose+Hands Egocentric Vision Testing")
    parser.add_argument("--dataset", required=True,
                       help="Path to egocentric dataset")
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard", "all"],
                       default="all", help="Difficulty level to test")
    parser.add_argument("--max-samples", type=int,
                       help="Maximum samples per difficulty level")
    parser.add_argument("--mediapipe-url", default="http://localhost:8000",
                       help="MediaPipe service URL")
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
    tester = MediaPipeEgocentricTester(
        dataset_path=dataset_path,
        cvat_url=args.cvat_url,
        mediapipe_service_url=args.mediapipe_url
    )

    # Run tests
    results = tester.run_comprehensive_test(
        difficulties=difficulties,
        max_samples_per_difficulty=args.max_samples
    )

    # Save results
    timestamp = int(time.time())
    result_file = output_dir / f"mediapipe_egocentric_test_{timestamp}.json"
    tester.save_results(results, result_file)

    # Print summary
    if "error" not in results:
        summary = results["summary"]
        print("\n" + "="*60)
        print("🎯 MEDIAPIPE POSE+HANDS EGOCENTRIC TESTING SUMMARY")
        print("="*60)
        print(f"📊 Images Tested: {summary['total_images_tested']}")
        print(f"✅ Successful: {summary['total_successful']}")
        print(f"❌ Failed: {summary['total_failed']}")
        print(".2f")
        print(".2f")
        print(".2f")
        print(".2f")
        print(".2f")
        print(".1f")
        print(".1f")
        print(".2f")
        print(f"📁 Results: {result_file}")
        print("="*60)

    return 0 if "error" not in results else 1

if __name__ == "__main__":
    exit(main())
