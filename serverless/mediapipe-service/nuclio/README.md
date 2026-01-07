# MediaPipe Nuclio Function

This directory contains the Nuclio serverless function that proxies requests to the MediaPipe service for pose and hand detection.

## Files

- `function.yaml` - Nuclio function configuration (automatically updated with skeleton spec)
- `main.py` - Function handler that proxies requests to MediaPipe service
- `deploy.sh` - Deployment script (automatically prepares function.yaml before deployment)
- `prepare_function_yaml.py` - Script to inject skeleton spec from JSON into function.yaml

## Deployment

### Automatic Deployment (Recommended)

The `deploy_egocentric_models.sh` script automatically:
1. Sets up and starts the MediaPipe service
2. Prepares `function.yaml` with skeleton spec
3. Deploys the Nuclio function

```bash
cd serverless
./deploy_egocentric_models.sh --mediapipe
```

### Manual Deployment

If deploying manually:

```bash
cd serverless/mediapipe-service/nuclio
./deploy.sh
```

The `deploy.sh` script automatically runs `prepare_function_yaml.py` to inject the skeleton spec from `../mediapipe-skeletons-raw-editor.json` into `function.yaml` before deployment.

## Skeleton Configuration

The skeleton configuration is stored in:
- `../mediapipe-skeletons-raw-editor.json` - Source of truth (contains both "person" and "hands" labels, used for CVAT dashboard import)

The `prepare_function_yaml.py` script automatically:
- Reads the skeleton JSON file (from `mediapipe-skeletons-raw-editor.json`)
- Extracts the "person" label (if multiple labels exist)
- Converts it to a JSON string
- Injects it into `function.yaml` with proper YAML escaping
- Uses PyYAML's built-in serializer for safe escaping

## Function Details

- **Name**: `pth-google-mediapipe-pose-hands`
- **Type**: Detector (Skeleton)
- **Keypoints**: 57 (33 body + 21 left hand + 21 right hand - 18 overlapping wrist points)
- **Service URL**: `http://mediapipe-pose:8000` (Docker container on cvat_cvat network)

## Notes

- The skeleton spec in `function.yaml` is automatically generated from the JSON file
- SVG content uses unescaped `<` and `>` characters (required for CVAT UI)
- Uses `data-label-name` (not `data-label-id`) for import compatibility
- The spec field is automatically escaped for YAML format using PyYAML's serializer
