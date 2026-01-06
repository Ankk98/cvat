#!/usr/bin/env python3
"""
Nuclio proxy function for FCAF3D 3D Object Detection Service
==========================================================

This function proxies requests from CVAT to the FCAF3D service running
on the host machine. It's optimized for 3D cuboid detection from point clouds.

The FCAF3D service should be running on port 8000 (default).
"""

import os
import json
import logging
import requests
from typing import Dict, Any

# Configure logging
# Note: Use context.logger in handler instead of this logger
logging.basicConfig(level=logging.INFO)

# Get FCAF3D service URL from environment
# Use container name for Docker network communication (maintainable and reliable)
# Container name 'fcaf3d-service' is accessible from any container on the same network
FCAF3D_SERVICE_URL = os.getenv("FCAF3D_SERVICE_URL", "http://fcaf3d-service:8000")
FCAF3D_SERVICE_TIMEOUT = int(os.getenv("FCAF3D_SERVICE_TIMEOUT", "60"))  # FCAF3D can be slower
DETECT_ENDPOINT = f"{FCAF3D_SERVICE_URL}/detect"

def handler(context, event):
    """
    Proxy handler for FCAF3D 3D object detection service.

    Args:
        context: Nuclio context object
        event: Nuclio event object containing the request

    Returns:
        Response from FCAF3D service (CVAT-compatible cuboid format)
    """
    try:
        # Parse request body
        # CVAT sends the payload directly in event.body
        data = event.body

        # Extract parameters from request
        # CVAT sends: {"image": "...", "frame": 0, "shapes": [...], "states": [...], "threshold": 0.3}
        image_data = data.get('image')
        threshold = data.get('threshold', 0.3)
        # CVAT uses 'frame', but we also support 'frame_number' for backwards compatibility/direct calls
        frame_number = data.get('frame', data.get('frame_number', 0))

        context.logger.info(f"Received request with keys: {list(data.keys()) if isinstance(data, dict) else type(data)}")

        if not image_data:
            context.logger.error("No image provided in request")
            return context.Response(
                body=json.dumps({'error': 'No image provided'}),
                headers={},
                content_type='application/json',
                status_code=400
            )

        # Prepare request to FCAF3D service
        # Forward all parameters for 3D detection
        fcaf3d_payload = {
            'image': image_data,
            'threshold': threshold,
            'frame_number': frame_number,
            'job_id': data.get('job_id'),
            'task_id': data.get('task_id'),
            'shapes': data.get('shapes'),
            'states': data.get('states')
        }

        context.logger.info(f"Forwarding request to FCAF3D service at {DETECT_ENDPOINT}")
        context.logger.info(f"Image data length: {len(image_data) if image_data else 0}, threshold: {threshold}, frame: {frame_number}")

        # Forward request to FCAF3D service
        try:
            response = requests.post(
                DETECT_ENDPOINT,
                json=fcaf3d_payload,
                timeout=FCAF3D_SERVICE_TIMEOUT,
                headers={'Content-Type': 'application/json'}
            )
            response.raise_for_status()

            # Get response data - FCAF3D service returns detections directly
            result = response.json()

            context.logger.info(f"FCAF3D service returned {len(result.get('detections', []))} detection(s)")

            # Return the detections directly (CVAT expects a list of detections)
            detections = result.get('detections', [])

            # Return the response directly as JSON
            return context.Response(
                body=json.dumps(detections),
                headers={},
                content_type='application/json',
                status_code=200
            )

        except requests.exceptions.ConnectionError as e:
            context.logger.error(f"Failed to connect to FCAF3D service at {DETECT_ENDPOINT}: {e}")
            context.logger.error("Make sure the FCAF3D service is running on port 8000")
            return context.Response(
                body=json.dumps({
                    'error': 'FCAF3D service unavailable',
                    'message': f'Cannot connect to {DETECT_ENDPOINT}. Ensure the service is running.'
                }),
                headers={},
                content_type='application/json',
                status_code=503
            )
        except requests.exceptions.Timeout as e:
            context.logger.error(f"Request to FCAF3D service timed out: {e}")
            return context.Response(
                body=json.dumps({
                    'error': 'FCAF3D service timeout',
                    'message': 'The request to FCAF3D service exceeded the timeout limit'
                }),
                headers={},
                content_type='application/json',
                status_code=504
            )
        except requests.exceptions.HTTPError as e:
            context.logger.error(f"FCAF3D service returned error: {e}")
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
