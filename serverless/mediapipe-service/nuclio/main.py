#!/usr/bin/env python3
"""
Nuclio proxy function for MediaPipe Pose + Hands Detection Service
==================================================================

This function proxies requests from CVAT to the MediaPipe service running
on the host machine. It's optimized for egocentric video pose detection.

The MediaPipe service should be running on port 8000 (default).
"""

import os
import json
import logging
import requests
from typing import Dict, Any

# Configure logging
# Note: Use context.logger in handler instead of this logger
logging.basicConfig(level=logging.INFO)

# Get MediaPipe service URL from environment
# Use container name for Docker network communication (maintainable and reliable)
# Container name 'mediapipe-pose' is accessible from any container on the same network
MEDIAPIPE_SERVICE_URL = os.getenv("MEDIAPIPE_SERVICE_URL", "http://mediapipe-pose:8000")
MEDIAPIPE_SERVICE_TIMEOUT = int(os.getenv("MEDIAPIPE_SERVICE_TIMEOUT", "30"))
DETECT_ENDPOINT = f"{MEDIAPIPE_SERVICE_URL}/detect"

def handler(context, event):
    """
    Proxy handler for MediaPipe pose detection service.

    Args:
        context: Nuclio context object
        event: Nuclio event object containing the request

    Returns:
        Response from MediaPipe service (CVAT-compatible skeleton format)
    """
    try:
        # Parse request body
        # CVAT sends the payload directly in event.body
        data = event.body

        # Extract image and threshold from request
        # CVAT sends: {"image": "base64_string", "threshold": 0.3}
        image_data = data.get('image')
        threshold = data.get('threshold', 0.3)

        context.logger.info(f"Received request with keys: {list(data.keys()) if isinstance(data, dict) else type(data)}")

        if not image_data:
            context.logger.error("No image provided in request")
            return context.Response(
                body=json.dumps({'error': 'No image provided'}),
                headers={},
                content_type='application/json',
                status_code=400
            )

        # Prepare request to MediaPipe service
        mediapipe_payload = {
            'image': image_data,
            'threshold': threshold
        }

        context.logger.info(f"Forwarding request to MediaPipe service at {DETECT_ENDPOINT}")
        context.logger.info(f"Image data length: {len(image_data) if image_data else 0}, threshold: {threshold}")

        # Forward request to MediaPipe service
        try:
            response = requests.post(
                DETECT_ENDPOINT,
                json=mediapipe_payload,
                timeout=MEDIAPIPE_SERVICE_TIMEOUT,
                headers={'Content-Type': 'application/json'}
            )
            response.raise_for_status()

            # Get response data - MediaPipe service returns a list of skeletons
            result = response.json()

            context.logger.info(f"MediaPipe service returned {len(result) if isinstance(result, list) else 'non-list'} result(s)")

            # Return the response directly as JSON (MediaPipe service already returns CVAT-compatible format)
            # CVAT expects a list of skeleton objects
            return context.Response(
                body=json.dumps(result),
                headers={},
                content_type='application/json',
                status_code=200
            )

        except requests.exceptions.ConnectionError as e:
            context.logger.error(f"Failed to connect to MediaPipe service at {DETECT_ENDPOINT}: {e}")
            context.logger.error("Make sure the MediaPipe service is running on port 8000")
            return context.Response(
                body=json.dumps({
                    'error': 'MediaPipe service unavailable',
                    'message': f'Cannot connect to {DETECT_ENDPOINT}. Ensure the service is running.'
                }),
                headers={},
                content_type='application/json',
                status_code=503
            )
        except requests.exceptions.Timeout as e:
            context.logger.error(f"Request to MediaPipe service timed out: {e}")
            return context.Response(
                body=json.dumps({
                    'error': 'MediaPipe service timeout',
                    'message': 'The request to MediaPipe service exceeded the timeout limit'
                }),
                headers={},
                content_type='application/json',
                status_code=504
            )
        except requests.exceptions.HTTPError as e:
            context.logger.error(f"MediaPipe service returned error: {e}")
            context.logger.error(f"Response status: {response.status_code}, Response text: {response.text[:500]}")
            try:
                error_body = response.json()
            except:
                error_body = {'error': str(e), 'message': response.text[:500]}
            # Return 503 for 5xx errors (service issues), 400 for 4xx (client errors)
            status_code = 503 if response.status_code >= 500 else response.status_code
            return context.Response(
                body=json.dumps(error_body),
                headers={},
                content_type='application/json',
                status_code=status_code
            )

    except json.JSONDecodeError as e:
        context.logger.error(f"Invalid JSON in request: {e}")
        return context.Response(
            body=json.dumps({'error': 'Invalid JSON in request body'}),
            headers={},
            content_type='application/json',
            status_code=400
        )
    except Exception as e:
        context.logger.error(f"Unexpected error in handler: {e}", exc_info=True)
        return context.Response(
            body=json.dumps({
                'error': 'Internal server error',
                'message': str(e)
            }),
            headers={},
            content_type='application/json',
            status_code=500
        )

