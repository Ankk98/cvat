#!/bin/bash
# Deploy MediaPipe Pose Nuclio Function for Egocentric Videos
# ============================================================
#
# This script deploys a Nuclio function that proxies requests to the
# MediaPipe service for egocentric video pose detection.
#
# Prerequisites:
#   1. MediaPipe service must be running on port 8000
#   2. Nuclio must be running and accessible
#   3. Docker must be running
#
# Usage:
#   ./deploy.sh
#
# To check if MediaPipe service is running:
#   curl http://localhost:8000/health

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
NUCLIO_DIR="$SCRIPT_DIR"
FUNCTION_NAME="pth-google-mediapipe-pose-hands"

echo "=========================================="
echo "Deploying MediaPipe Pose Nuclio Function"
echo "=========================================="
echo ""

# Check if MediaPipe service is running
echo "Checking MediaPipe service status..."
if curl -s -f http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ MediaPipe service is running on port 8000"
else
    echo "⚠️  WARNING: MediaPipe service is not responding on port 8000"
    echo "   Please start the service first:"
    echo "   cd $(dirname "$SCRIPT_DIR") && ./start.sh"
    echo ""
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check if Nuclio project exists, create if not
echo ""
echo "Checking Nuclio project..."
if nuctl get project cvat --platform local > /dev/null 2>&1; then
    echo "✅ Nuclio project 'cvat' exists"
else
    echo "Creating Nuclio project 'cvat'..."
    nuctl create project cvat --platform local
fi

# Deploy the function
echo ""
echo "Deploying Nuclio function: $FUNCTION_NAME"
echo "Path: $NUCLIO_DIR"
echo ""

nuctl deploy \
    --project-name cvat \
    --path "$NUCLIO_DIR" \
    --file "$NUCLIO_DIR/function.yaml" \
    --platform local \
    --env CVAT_FUNCTIONS_REDIS_HOST=cvat_redis_ondisk \
    --env CVAT_FUNCTIONS_REDIS_PORT=6666 \
    --platform-config '{"attributes": {"network": "cvat_cvat"}}'

echo ""
echo "=========================================="
echo "✅ Deployment completed!"
echo "=========================================="
echo ""
echo "Function name: $FUNCTION_NAME"
echo ""
echo "To check function status:"
echo "  nuctl get function $FUNCTION_NAME --platform local"
echo ""
echo "To view function logs:"
echo "  nuctl get function $FUNCTION_NAME --platform local | grep -A 10 'Status'"
echo ""
echo "To test the function:"
echo "  curl -X POST http://localhost:<port>/ -H 'Content-Type: application/json' -d '{\"image\": \"base64_image_data\", \"threshold\": 0.3}'"
echo ""
echo "Note: The function will proxy requests to MediaPipe service at http://host.docker.internal:8000"
echo ""

