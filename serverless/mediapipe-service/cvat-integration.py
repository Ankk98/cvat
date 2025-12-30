#!/usr/bin/env python3
"""
CVAT Integration Script for MediaPipe Pose Service
==================================================

This script automatically configures CVAT to use the MediaPipe pose detection service.

Features:
- Automatic function registration
- Health checks
- Test image processing
- Configuration validation

Usage:
    python cvat-integration.py --cvat-url http://localhost:8080 --service-url http://localhost:8000

Requirements:
- requests
- cvat-sdk (optional, for advanced integration)
"""

import argparse
import base64
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Optional

import requests
from PIL import Image, ImageDraw
import io

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class CVATIntegration:
    """CVAT integration for MediaPipe pose service."""

    def __init__(self, cvat_url: str, service_url: str, auth_token: Optional[str] = None):
        self.cvat_url = cvat_url.rstrip('/')
        self.service_url = service_url.rstrip('/')
        self.auth_token = auth_token
        self.session = requests.Session()

        if auth_token:
            self.session.headers.update({'Authorization': f'Token {auth_token}'})

    def test_service_health(self) -> bool:
        """Test if the MediaPipe service is healthy."""
        try:
            response = self.session.get(f"{self.service_url}/health", timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'healthy':
                    logger.info("✅ MediaPipe service is healthy")
                    return True
            logger.error(f"❌ MediaPipe service health check failed: {response.status_code}")
            return False
        except Exception as e:
            logger.error(f"❌ Cannot connect to MediaPipe service: {e}")
            return False

    def create_test_image(self) -> str:
        """Create a simple test image with a stick figure for testing."""
        # Create a simple image with a basic pose
        img = Image.new('RGB', (400, 600), color='white')
        draw = ImageDraw.Draw(img)

        # Draw a simple stick figure pose
        # Head
        draw.ellipse([175, 50, 225, 100], fill='lightblue', outline='black')
        # Body
        draw.rectangle([195, 100, 205, 200], fill='black')
        # Arms
        draw.rectangle([150, 120, 195, 130], fill='black')  # left arm
        draw.rectangle([205, 120, 250, 130], fill='black')  # right arm
        # Hands
        draw.ellipse([140, 115, 155, 130], fill='pink')     # left hand
        draw.ellipse([240, 115, 255, 130], fill='pink')     # right hand
        # Legs
        draw.rectangle([195, 200, 205, 300], fill='black')  # left leg
        draw.rectangle([195, 200, 205, 300], fill='black')  # right leg
        # Feet
        draw.ellipse([185, 290, 200, 305], fill='brown')    # left foot
        draw.ellipse([200, 290, 215, 305], fill='brown')    # right foot

        # Convert to base64
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG')
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

        return img_base64

    def test_pose_detection(self) -> bool:
        """Test pose detection with a sample image."""
        try:
            # Create test image
            test_image = self.create_test_image()

            # Send to service
            response = self.session.post(
                f"{self.service_url}/detect",
                json={"image": test_image, "threshold": 0.1},  # Low threshold for testing
                timeout=30
            )

            if response.status_code == 200:
                results = response.json()
                if isinstance(results, list) and len(results) > 0:
                    logger.info(f"✅ Pose detection working! Found {len(results)} poses")
                    for i, pose in enumerate(results):
                        elements = pose.get('elements', [])
                        logger.info(f"  Pose {i+1}: {len(elements)} keypoints detected")
                    return True
                else:
                    logger.warning("⚠️  Pose detection returned empty results")
                    return False
            else:
                logger.error(f"❌ Pose detection failed: {response.status_code} - {response.text}")
                return False

        except Exception as e:
            logger.error(f"❌ Pose detection test failed: {e}")
            return False

    def get_cvat_functions_endpoint(self) -> Optional[str]:
        """Get CVAT functions endpoint."""
        # Try different possible endpoints
        endpoints = [
            f"{self.cvat_url}/api/functions",
            f"{self.cvat_url}/functions"
        ]

        for endpoint in endpoints:
            try:
                response = self.session.get(endpoint, timeout=5)
                if response.status_code in [200, 401, 403]:  # 401/403 means endpoint exists but needs auth
                    return endpoint
            except:
                continue

        return None

    def register_function(self) -> bool:
        """Register the MediaPipe function with CVAT."""
        functions_endpoint = self.get_cvat_functions_endpoint()

        if not functions_endpoint:
            logger.error("❌ Cannot find CVAT functions endpoint")
            return False

        # Function configuration for CVAT
        function_config = {
            "name": "MediaPipe Pose Detection",
            "description": "33-point pose estimation optimized for egocentric vision",
            "type": "detector",
            "url": f"{self.service_url}/detect",
            "method": "POST",
            "headers": {
                "Content-Type": "application/json"
            },
            "parameters": {
                "threshold": {
                    "type": "number",
                    "default": 0.3,
                    "min": 0.0,
                    "max": 1.0,
                    "description": "Confidence threshold for pose keypoints"
                }
            },
            "spec": {
                "labels": [
                    {
                        "name": "person-skeleton",
                        "type": "skeleton",
                        "svg": "<circle r=\"2\" cx=\"50\" cy=\"20\" data-type=\"element node\" data-element-id=\"0\" data-node-id=\"0\" data-label-name=\"nose\"></circle><circle r=\"2\" cx=\"35\" cy=\"15\" data-type=\"element node\" data-element-id=\"1\" data-node-id=\"1\" data-label-name=\"left_eye\"></circle><circle r=\"2\" cx=\"65\" cy=\"15\" data-type=\"element node\" data-element-id=\"2\" data-node-id=\"2\" data-label-name=\"right_eye\"></circle><circle r=\"2\" cx=\"25\" cy=\"25\" data-type=\"element node\" data-element-id=\"3\" data-node-id=\"3\" data-label-name=\"left_ear\"></circle><circle r=\"2\" cx=\"75\" cy=\"25\" data-type=\"element node\" data-element-id=\"4\" data-node-id=\"4\" data-label-name=\"right_ear\"></circle><circle r=\"2\" cx=\"50\" cy=\"50\" data-type=\"element node\" data-element-id=\"5\" data-node-id=\"5\" data-label-name=\"left_shoulder\"></circle><circle r=\"2\" cx=\"50\" cy=\"70\" data-type=\"element node\" data-element-id=\"6\" data-node-id=\"6\" data-label-name=\"right_shoulder\"></circle><circle r=\"2\" cx=\"30\" cy=\"70\" data-type=\"element node\" data-element-id=\"7\" data-node-id=\"7\" data-label-name=\"left_elbow\"></circle><circle r=\"2\" cx=\"70\" cy=\"90\" data-type=\"element node\" data-element-id=\"8\" data-node-id=\"8\" data-label-name=\"right_elbow\"></circle><circle r=\"2\" cx=\"20\" cy=\"90\" data-type=\"element node\" data-element-id=\"9\" data-node-id=\"9\" data-label-name=\"left_wrist\"></circle><circle r=\"2\" cx=\"80\" cy=\"110\" data-type=\"element node\" data-element-id=\"10\" data-node-id=\"10\" data-label-name=\"right_wrist\"></circle><circle r=\"2\" cx=\"45\" cy=\"85\" data-type=\"element node\" data-element-id=\"11\" data-node-id=\"11\" data-label-name=\"left_hip\"></circle><circle r=\"2\" cx=\"55\" cy=\"85\" data-type=\"element node\" data-element-id=\"12\" data-node-id=\"12\" data-label-name=\"right_hip\"></circle><circle r=\"2\" cx=\"40\" cy=\"110\" data-type=\"element node\" data-element-id=\"13\" data-node-id=\"13\" data-label-name=\"left_knee\"></circle><circle r=\"2\" cx=\"60\" cy=\"130\" data-type=\"element node\" data-element-id=\"14\" data-node-id=\"14\" data-label-name=\"right_knee\"></circle><circle r=\"2\" cx=\"35\" cy=\"130\" data-type=\"element node\" data-element-id=\"15\" data-node-id=\"15\" data-label-name=\"left_ankle\"></circle><circle r=\"2\" cx=\"65\" cy=\"150\" data-type=\"element node\" data-element-id=\"16\" data-node-id=\"16\" data-label-name=\"right_ankle\"></circle>",
                        "sublabels": [
                            {"id": 0, "name": "nose", "type": "points"},
                            {"id": 1, "name": "left_eye", "type": "points"},
                            {"id": 2, "name": "right_eye", "type": "points"},
                            {"id": 3, "name": "left_ear", "type": "points"},
                            {"id": 4, "name": "right_ear", "type": "points"},
                            {"id": 5, "name": "left_shoulder", "type": "points"},
                            {"id": 6, "name": "right_shoulder", "type": "points"},
                            {"id": 7, "name": "left_elbow", "type": "points"},
                            {"id": 8, "name": "right_elbow", "type": "points"},
                            {"id": 9, "name": "left_wrist", "type": "points"},
                            {"id": 10, "name": "right_wrist", "type": "points"},
                            {"id": 11, "name": "left_hip", "type": "points"},
                            {"id": 12, "name": "right_hip", "type": "points"},
                            {"id": 13, "name": "left_knee", "type": "points"},
                            {"id": 14, "name": "right_knee", "type": "points"},
                            {"id": 15, "name": "left_ankle", "type": "points"},
                            {"id": 16, "name": "right_ankle", "type": "points"}
                        ]
                    }
                ]
            }
        }

        try:
            response = self.session.post(
                functions_endpoint,
                json=function_config,
                timeout=30
            )

            if response.status_code in [200, 201]:
                logger.info("✅ MediaPipe function registered successfully with CVAT")
                return True
            elif response.status_code == 401:
                logger.error("❌ CVAT authentication required. Please provide auth token.")
                logger.info("To get auth token: CVAT → Settings → Account → Token")
                return False
            else:
                logger.error(f"❌ Failed to register function: {response.status_code} - {response.text}")
                return False

        except Exception as e:
            logger.error(f"❌ Error registering function: {e}")
            return False

    def run_integration(self) -> bool:
        """Run the complete integration process."""
        logger.info("Starting CVAT integration for MediaPipe Pose Service")
        logger.info("=" * 60)

        # Step 1: Test service health
        logger.info("\n1. Testing MediaPipe service health...")
        if not self.test_service_health():
            logger.error("❌ Service health check failed. Please start the service first.")
            return False

        # Step 2: Test pose detection
        logger.info("\n2. Testing pose detection...")
        if not self.test_pose_detection():
            logger.error("❌ Pose detection test failed.")
            return False

        # Step 3: Register with CVAT
        logger.info("\n3. Registering function with CVAT...")
        if not self.register_function():
            logger.error("❌ CVAT registration failed.")
            return False

        logger.info("\n" + "=" * 60)
        logger.info("✅ CVAT integration completed successfully!")
        logger.info("")
        logger.info("You can now use MediaPipe pose detection in CVAT:")
        logger.info("  - Go to a task → Draw new shape → Select 'MediaPipe Pose Detection'")
        logger.info("  - Or use the function in auto-annotation workflows")

        return True

def main():
    parser = argparse.ArgumentParser(description="CVAT Integration for MediaPipe Pose Service")
    parser.add_argument("--cvat-url", required=True, help="CVAT server URL (e.g., http://localhost:8080)")
    parser.add_argument("--service-url", required=True, help="MediaPipe service URL (e.g., http://localhost:8000)")
    parser.add_argument("--auth-token", help="CVAT authentication token (optional)")
    parser.add_argument("--test-only", action="store_true", help="Only test the service, don't register with CVAT")

    args = parser.parse_args()

    # Create integration instance
    integration = CVATIntegration(args.cvat_url, args.service_url, args.auth_token)

    if args.test_only:
        logger.info("Running service tests only...")
        success = integration.test_service_health() and integration.test_pose_detection()
    else:
        success = integration.run_integration()

    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
