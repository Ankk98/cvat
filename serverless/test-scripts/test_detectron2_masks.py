#!/usr/bin/env python3
"""
Test Detectron2 Mask Output Format
===================================

Tests Detectron2 Mask R-CNN function to verify it returns masks (not boxes) in correct CVAT format.

Mask Format:
    - Service outputs: [pixel1, pixel2, ..., x_min, y_min, x_max, y_max] (flattened pixels)
    - CVAT backend converts to RLE automatically
    - Matches OpenVINO to_cvat_mask() format

Usage:
    python test_detectron2_masks.py [--url URL] [--image PATH]
"""

import argparse
import base64
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import requests
from PIL import Image

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_detectron2_masks(
    detectron_url: str = "http://localhost:32957",
    image_path: Optional[str] = None,
    threshold: float = 0.3
) -> Dict:
    """Test Detectron2 function and verify mask output format."""

    # Load or create test image
    if image_path and Path(image_path).exists():
        with open(image_path, 'rb') as f:
            image_data = f.read()
        img = Image.open(Path(image_path))
        logger.info(f"Loaded image: {image_path} ({img.size[0]}x{img.size[1]})")
    else:
        # Create a test image with objects
        img = Image.new('RGB', (640, 480), color='white')
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)
        # Draw colored rectangles (simulating objects)
        draw.rectangle([100, 100, 200, 200], fill='red')
        draw.rectangle([300, 150, 400, 250], fill='blue')
        draw.rectangle([500, 200, 600, 300], fill='green')
        buffer = img.tobytes()
        image_data = buffer
        logger.info("Created test image with colored rectangles")

    # Convert to base64
    if isinstance(image_data, bytes):
        img_base64 = base64.b64encode(image_data).decode('utf-8')
    else:
        buffer = image_data
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

    # Call Detectron2 service
    logger.info(f"Calling Detectron2 service at {detectron_url}...")
    payload = {"image": img_base64, "threshold": threshold}

    try:
        response = requests.post(detectron_url, json=payload, timeout=60)
        response.raise_for_status()
        results = response.json()

        logger.info(f"✅ Service responded successfully")
        logger.info(f"Number of detections: {len(results)}")

        # Analyze results
        analysis = {
            "total_detections": len(results),
            "mask_detections": 0,
            "box_detections": 0,
            "detections": []
        }

        for i, result in enumerate(results):
            det_type = result.get("type", "unknown")
            label = result.get("label", "unknown")
            confidence = result.get("confidence", "0")
            points = result.get("points", [])

            detection_info = {
                "index": i,
                "type": det_type,
                "label": label,
                "confidence": confidence,
                "points_length": len(points),
                "has_mask_field": "mask" in result,
            }

            if det_type == "mask":
                analysis["mask_detections"] += 1
                # Verify flattened pixels format (CVAT expects this, backend converts to RLE)
                # Format: [pixel1, pixel2, ..., x_min, y_min, x_max, y_max]
                if len(points) >= 6:  # Minimum: 2 pixels + 4 bbox coordinates
                    # Last 4 should be bbox: [x_min, y_min, x_max, y_max]
                    bbox = points[-4:]
                    pixel_data_length = len(points) - 4
                    detection_info["pixel_data_length"] = pixel_data_length
                    detection_info["bbox"] = bbox
                    detection_info["bbox_format"] = "[x_min, y_min, x_max, y_max]"
                    detection_info["format"] = "flattened_pixels"

                    # Check if format looks valid (should have pixel data)
                    if pixel_data_length > 0:
                        detection_info["format_valid"] = True
                        detection_info["first_10_pixels"] = points[:10]
                        # Check if values are 0/1 (binary mask pixels)
                        pixel_values = points[:-4]
                        unique_values = set(pixel_values[:100])  # Sample first 100
                        if unique_values.issubset({0, 1}):
                            detection_info["pixel_format_valid"] = True
                        else:
                            detection_info["pixel_format_valid"] = False
                            logger.warning(f"⚠️  Detection {i} has non-binary pixel values: {unique_values}")
                    else:
                        detection_info["format_valid"] = False
                        logger.warning(f"⚠️  Detection {i} has mask type but no pixel data!")
                else:
                    detection_info["format_valid"] = False
                    logger.error(f"❌ Detection {i} has mask type but invalid points format (length {len(points)}, need >= 6)!")
            elif det_type == "rectangle":
                analysis["box_detections"] += 1
                detection_info["bbox"] = points
                logger.warning(f"⚠️  Detection {i} is rectangle type (expected mask)")
            else:
                logger.warning(f"⚠️  Detection {i} has unknown type: {det_type}")

            analysis["detections"].append(detection_info)

        # Summary
        logger.info("\n" + "="*60)
        logger.info("DETECTION SUMMARY")
        logger.info("="*60)
        logger.info(f"Total detections: {analysis['total_detections']}")
        logger.info(f"Mask detections: {analysis['mask_detections']}")
        logger.info(f"Box detections: {analysis['box_detections']}")

        if analysis['mask_detections'] > 0:
            logger.info("\n✅ Mask detections found!")
            for det in analysis['detections']:
                if det['type'] == 'mask':
                    logger.info(f"  - {det['label']}: {det['confidence']} confidence")
                    logger.info(f"    Pixel data length: {det.get('pixel_data_length', 0)}, Bbox: {det.get('bbox', [])}")
                    if det.get('pixel_format_valid'):
                        logger.info(f"    ✅ Format valid (flattened pixels, binary values)")
                    else:
                        logger.warning(f"    ⚠️  Format issue detected")
        else:
            logger.warning("\n⚠️  No mask detections found!")
            if analysis['box_detections'] > 0:
                logger.error("❌ Function is returning boxes instead of masks!")
            else:
                logger.info("   (No detections at all - might be threshold or image issue)")

        return analysis

    except requests.exceptions.RequestException as e:
        logger.error(f"❌ Failed to call Detectron2 service: {e}")
        return {"error": str(e)}
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Test Detectron2 mask output format")
    parser.add_argument("--url", default="http://localhost:32957",
                       help="Detectron2 service URL")
    parser.add_argument("--image", type=str, default=None,
                       help="Path to test image (optional)")
    parser.add_argument("--threshold", type=float, default=0.3,
                       help="Detection threshold")

    args = parser.parse_args()

    # Check service availability
    try:
        response = requests.get(f"{args.url}/health", timeout=5)
        if response.status_code == 200:
            logger.info("✅ Service is healthy")
        else:
            logger.warning(f"⚠️  Service health check returned {response.status_code}")
    except:
        logger.warning("⚠️  Could not check service health (continuing anyway)")

    # Run test
    result = test_detectron2_masks(args.url, args.image, args.threshold)

    # Save results
    output_file = Path("test_detectron2_masks_results.json")
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    logger.info(f"\n📄 Results saved to: {output_file}")

    # Exit code
    if result.get("error"):
        sys.exit(1)
    elif result.get("mask_detections", 0) == 0 and result.get("box_detections", 0) > 0:
        logger.error("\n❌ TEST FAILED: Function returns boxes instead of masks!")
        sys.exit(1)
    elif result.get("mask_detections", 0) > 0:
        logger.info("\n✅ TEST PASSED: Function returns masks correctly!")
        sys.exit(0)
    else:
        logger.warning("\n⚠️  TEST INCONCLUSIVE: No detections found")
        sys.exit(0)


if __name__ == "__main__":
    main()

