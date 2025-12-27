#!/usr/bin/env python3
"""
Test Egocentric Models with Real Downloaded Images
=================================================

Simple testing script for models using real images downloaded from the internet.
"""

import argparse
import json
import base64
import requests
import time
from pathlib import Path
from typing import Dict, List
from PIL import Image
import io

def test_sam_model(image_path: Path, sam_url: str) -> Dict:
    """Test SAM model on a single image."""
    try:
        # Load and encode image
        with open(image_path, 'rb') as f:
            img_data = f.read()

        img_base64 = base64.b64encode(img_data).decode('utf-8')

        # Test SAM
        payload = {"image": img_base64}
        start_time = time.time()
        response = requests.post(sam_url, json=payload, timeout=30)
        end_time = time.time()

        result = {
            "image": str(image_path.name),
            "success": response.status_code == 200,
            "response_time": end_time - start_time,
            "status_code": response.status_code,
            "error": None if response.status_code == 200 else response.text
        }

        if response.status_code == 200:
            try:
                response_data = response.json()
                result["response_size"] = len(str(response_data))
            except:
                result["response_size"] = 0

        return result

    except Exception as e:
        return {
            "image": str(image_path.name),
            "success": False,
            "response_time": 0,
            "status_code": 0,
            "error": str(e)
        }

def test_detectron2_model(image_path: Path, detectron_url: str) -> Dict:
    """Test Detectron2 model on a single image."""
    try:
        # Load and encode image
        with open(image_path, 'rb') as f:
            img_data = f.read()

        img_base64 = base64.b64encode(img_data).decode('utf-8')

        # Test Detectron2
        payload = {"image": img_base64, "threshold": 0.1}
        start_time = time.time()
        response = requests.post(detectron_url, json=payload, timeout=30)
        end_time = time.time()

        result = {
            "image": str(image_path.name),
            "success": response.status_code == 200,
            "response_time": end_time - start_time,
            "status_code": response.status_code,
            "error": None if response.status_code == 200 else response.text,
            "detections": 0
        }

        if response.status_code == 200:
            try:
                response_data = response.json()
                result["response_size"] = len(str(response_data))
                # Count detections if available
                if isinstance(response_data, list):
                    result["detections"] = len(response_data)
                elif isinstance(response_data, dict) and "detections" in response_data:
                    result["detections"] = len(response_data["detections"])
            except:
                result["response_size"] = 0

        return result

    except Exception as e:
        return {
            "image": str(image_path.name),
            "success": False,
            "response_time": 0,
            "status_code": 0,
            "error": str(e),
            "detections": 0
        }

def test_mediapipe_model(image_path: Path, mediapipe_url: str) -> Dict:
    """Test MediaPipe model on a single image."""
    try:
        # Load and encode image
        with open(image_path, 'rb') as f:
            img_data = f.read()

        img_base64 = base64.b64encode(img_data).decode('utf-8')

        # Test MediaPipe
        payload = {"image": img_base64}
        start_time = time.time()
        response = requests.post(f"{mediapipe_url}/detect", json=payload, timeout=30)
        end_time = time.time()

        result = {
            "image": str(image_path.name),
            "success": response.status_code == 200,
            "response_time": end_time - start_time,
            "status_code": response.status_code,
            "error": None if response.status_code == 200 else response.text,
            "pose_keypoints": 0,
            "hand_keypoints": 0
        }

        if response.status_code == 200:
            try:
                response_data = response.json()
                result["response_size"] = len(str(response_data))
                # Count keypoints for MediaPipe response format
                if isinstance(response_data, list) and response_data:
                    # MediaPipe returns list of detections
                    detection = response_data[0]  # Take first detection
                    if "elements" in detection:
                        keypoints = detection["elements"]
                        # Count pose keypoints (body joints)
                        pose_keypoints = [k for k in keypoints if not k["label"].startswith(("left_", "right_"))]
                        hand_keypoints = [k for k in keypoints if k["label"].startswith(("left_", "right_"))]
                        result["pose_keypoints"] = len(pose_keypoints)
                        result["hand_keypoints"] = len(hand_keypoints)
            except:
                result["response_size"] = 0

        return result

    except Exception as e:
        return {
            "image": str(image_path.name),
            "success": False,
            "response_time": 0,
            "status_code": 0,
            "error": str(e),
            "pose_keypoints": 0,
            "hand_keypoints": 0
        }

def find_images_in_directory(directory: Path) -> List[Path]:
    """Find all image files in a directory."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    return [
        f for f in directory.iterdir()
        if f.is_file() and f.suffix.lower() in image_extensions
    ]

def run_tests(dataset_path: Path, model: str, service_url: str, max_samples: int = 5):
    """Run tests for a specific model and dataset."""
    print(f"🧪 Testing {model} on {dataset_path.name}")
    print("-" * 50)

    # Find images
    images = find_images_in_directory(dataset_path)
    if not images:
        print(f"❌ No images found in {dataset_path}")
        return None

    print(f"📸 Found {len(images)} images, testing {min(max_samples, len(images))}...")

    results = []
    test_func = {
        "sam": test_sam_model,
        "detectron2": test_detectron2_model,
        "mediapipe": test_mediapipe_model
    }.get(model.lower())

    if not test_func:
        print(f"❌ Unknown model: {model}")
        return None

    # Test images
    successful = 0
    total_time = 0

    for i, image_path in enumerate(images[:max_samples]):
        print(f"Testing {i+1}/{min(max_samples, len(images))}: {image_path.name}")
        result = test_func(image_path, service_url)
        results.append(result)

        if result["success"]:
            successful += 1
        total_time += result["response_time"]

        # Small delay to be nice to services
        time.sleep(0.5)

    # Summary
    success_rate = successful / len(results) if results else 0
    avg_time = total_time / len(results) if results else 0

    summary = {
        "model": model,
        "dataset": str(dataset_path.name),
        "total_images_tested": len(results),
        "successful_inferences": successful,
        "failed_inferences": len(results) - successful,
        "success_rate": success_rate,
        "average_response_time": avg_time,
        "results": results
    }

    print("\n📊 Results:")
    print(f"   Images tested: {len(results)}")
    print(f"   Successful: {successful}")
    print(".1%")
    print(".2f")

    return summary

def main():
    parser = argparse.ArgumentParser(description="Test egocentric models with real images")
    parser.add_argument("--dataset", required=True, help="Path to dataset directory")
    parser.add_argument("--model", choices=["sam", "detectron2", "mediapipe"], required=True,
                       help="Model to test")
    parser.add_argument("--url", required=True, help="Service URL")
    parser.add_argument("--max-samples", type=int, default=5,
                       help="Maximum number of samples to test")
    parser.add_argument("--output", help="Output JSON file")

    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"❌ Dataset path not found: {dataset_path}")
        return 1

    print("🚀 Testing Egocentric Models with Real Images")
    print("=" * 50)

    # Run tests
    summary = run_tests(dataset_path, args.model, args.url, args.max_samples)

    if summary:
        # Save results
        output_file = args.output or f"test-results/{args.model}_real_images_test_{int(time.time())}.json"
        Path(output_file).parent.mkdir(exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\n📄 Results saved to: {output_file}")

        print("\n✅ Testing completed!")
        return 0
    else:
        return 1

if __name__ == "__main__":
    exit(main())
