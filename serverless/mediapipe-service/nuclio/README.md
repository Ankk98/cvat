# MediaPipe Pose Nuclio Function for Egocentric Videos

This Nuclio function acts as a proxy to the MediaPipe service, enabling CVAT to use MediaPipe pose and hand detection for egocentric videos through the Nuclio framework.

## Prerequisites

1. **MediaPipe service must be running** on port 8000
   ```bash
   cd /home/ankk98/repos/cvat/serverless/mediapipe-service
   ./start.sh
   ```

2. **Nuclio must be running** (usually via docker-compose)
   ```bash
   # Check if Nuclio is running
   docker ps | grep nuclio
   ```

3. **Docker must be running**

## Quick Start

### Option 1: Using the deployment script (Recommended)

```bash
cd /home/ankk98/repos/cvat/serverless/mediapipe-service/nuclio
./deploy.sh
```

### Option 2: Manual deployment

```bash
cd /home/ankk98/repos/cvat/serverless/mediapipe-service/nuclio

# Create Nuclio project if it doesn't exist
nuctl create project cvat --platform local

# Deploy the function
nuctl deploy \
    --project-name cvat \
    --path . \
    --file function.yaml \
    --platform local \
    --env CVAT_FUNCTIONS_REDIS_HOST=cvat_redis_ondisk \
    --env CVAT_FUNCTIONS_REDIS_PORT=6666 \
    --platform-config '{"attributes": {"network": "cvat_cvat"}}'
```

## Verify Deployment

### Check function status
```bash
nuctl get function pth-google-mediapipe-pose-hands --platform local
```

### Check function logs
```bash
nuctl get function pth-google-mediapipe-pose-hands --platform local | grep -A 10 'Status'
```

### Test the function
```bash
# Get the function port from the status output, then:
curl -X POST http://localhost:<port>/ \
  -H 'Content-Type: application/json' \
  -d '{"image": "base64_encoded_image", "threshold": 0.3}'
```

## Function Details

- **Function Name**: `pth-google-mediapipe-pose-hands`
- **Type**: Detector (skeleton)
- **Proxy Target**: `http://host.docker.internal:8000/detect`
- **Timeout**: 60 seconds
- **Max Request Size**: 32MB

## Configuration

The function uses environment variables to configure the MediaPipe service connection:

- `MEDIAPIPE_SERVICE_URL`: MediaPipe service URL (default: `http://host.docker.internal:8000`)
- `MEDIAPIPE_SERVICE_TIMEOUT`: Request timeout in seconds (default: `30`)

These can be overridden in the `function.yaml` file or during deployment.

## Troubleshooting

### MediaPipe service not accessible

If the function cannot connect to the MediaPipe service:

1. **Check if MediaPipe service is running:**
   ```bash
   curl http://localhost:8000/health
   ```

2. **Check Docker network:**
   The function uses `host.docker.internal` to access the host machine. If this doesn't work:
   - On Linux: Ensure Docker version 20.10+ is installed
   - Try using `172.17.0.1` instead of `host.docker.internal`
   - Or use the host's actual IP address

3. **Check function logs:**
   ```bash
   nuctl get function pth-google-mediapipe-pose-hands --platform local
   ```

### Function deployment fails

1. **Check Nuclio is running:**
   ```bash
   docker ps | grep nuclio
   ```

2. **Check Docker build:**
   The function builds a Docker image. Ensure Docker has enough resources.

3. **Check network configuration:**
   Ensure the function can access the `cvat_cvat` network.

## Usage in CVAT

Once deployed, the function will appear in CVAT's auto-annotation dropdown as:
- **Name**: MediaPipe Pose + Hands (Egocentric)
- **Type**: Detector
- **Skeleton**: 57 keypoints (17 body + 40 hand keypoints)

## Stopping the Function

```bash
nuctl delete function pth-google-mediapipe-pose-hands --platform local
```

## Related Files

- `function.yaml`: Nuclio function configuration
- `main.py`: Proxy handler implementation
- `deploy.sh`: Deployment script

