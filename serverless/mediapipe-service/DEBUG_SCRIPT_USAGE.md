# Debug Script Usage Guide

## Overview

The `debug_annotations.py` script compares MediaPipe annotations with CVAT annotations to identify issues with detection rate, coordinate accuracy, and keypoint positions.

## Authentication

The script requires authentication to access CVAT's API. You have three options:

### Option 1: Personal Access Token (Recommended)

Create a Personal Access Token in CVAT:
1. Go to CVAT UI → User Settings → Access Tokens
2. Create a new token
3. Use it with the script:

```bash
python debug_annotations.py \
    --cvat-url http://localhost:8080 \
    --task-id 14 \
    --job-id 10 \
    --token YOUR_TOKEN_HERE \
    --start-frame 0 \
    --end-frame 100
```

Or set environment variable:
```bash
export CVAT_ACCESS_TOKEN=YOUR_TOKEN_HERE
python debug_annotations.py --task-id 14 --job-id 10
```

### Option 2: Username/Password

```bash
python debug_annotations.py \
    --cvat-url http://localhost:8080 \
    --task-id 14 \
    --job-id 10 \
    --username admin \
    --password admin \
    --start-frame 0 \
    --end-frame 100
```

Or set environment variables:
```bash
export CVAT_USERNAME=admin
export CVAT_PASSWORD=admin
python debug_annotations.py --task-id 14 --job-id 10
```

### Option 3: Environment Variables Only

```bash
export CVAT_ACCESS_TOKEN=YOUR_TOKEN
# or
export CVAT_USERNAME=admin
export CVAT_PASSWORD=admin

python debug_annotations.py \
    --cvat-url http://localhost:8080 \
    --task-id 14 \
    --job-id 10 \
    --start-frame 0 \
    --end-frame 100 \
    --output-dir debug_results
```

## Command Line Arguments

```
--cvat-url          CVAT server URL (default: http://localhost:8080)
--task-id           CVAT task ID (required)
--job-id            CVAT job ID (optional, if not provided uses task)
--mediapipe-url     MediaPipe service URL (default: http://localhost:8000)
--start-frame       Start frame number (default: 0)
--end-frame         End frame number (default: 100)
--step              Frame step (default: 1)
--output-dir        Output directory for results (default: debug_output)
--no-visualize      Skip visualization generation
--username          CVAT username (for authentication)
--password          CVAT password (for authentication)
--token             CVAT Personal Access Token (for authentication)
```

## Example Usage

### Basic Analysis

```bash
cd serverless/mediapipe-service
python debug_annotations.py \
    --task-id 14 \
    --job-id 10 \
    --token YOUR_TOKEN \
    --start-frame 0 \
    --end-frame 50
```

### Full Analysis with Custom Output

```bash
python debug_annotations.py \
    --cvat-url http://localhost:8080 \
    --task-id 14 \
    --job-id 10 \
    --username admin \
    --password admin \
    --start-frame 0 \
    --end-frame 100 \
    --step 5 \
    --output-dir my_debug_results
```

### Without Visualization (Faster)

```bash
python debug_annotations.py \
    --task-id 14 \
    --job-id 10 \
    --token YOUR_TOKEN \
    --no-visualize
```

## Output

The script generates:

1. **`analysis_results.json`**: Detailed analysis with:
   - Statistics (detection rate, accuracy metrics)
   - Per-frame comparisons
   - Coordinate errors
   - Missing/extra detections

2. **`frame_XXXXXX_comparison.jpg`**: Visualization images showing:
   - CVAT annotations (green)
   - MediaPipe annotations (red)
   - Lines connecting matching keypoints
   - Yellow lines for large errors (>10% of image size)
   - Magenta lines for small errors

## Interpreting Results

### Detection Rate

Check `analysis_results.json` → `statistics`:
- `frames_with_cvat_annotations`: Frames that have manual annotations
- `frames_with_mediapipe_detections`: Frames where MediaPipe detected something
- `frames_with_both`: Frames with both CVAT and MediaPipe annotations

**Good:** `frames_with_both` / `frames_with_cvat_annotations` > 0.6 (60%)

### Coordinate Accuracy

Check `coordinate_errors` array:
- `normalized_distance`: Error as fraction of image size
- **Good:** < 0.05 (5% of image size)
- **Acceptable:** < 0.10 (10% of image size)
- **Bad:** > 0.10 (10% of image size)

### Missing Detections

Check `missing_detections` array:
- Lists frames where CVAT has annotations but MediaPipe doesn't
- **Issue:** Many missing detections = detection rate too low

### False Detections

Check `false_detections` array:
- Lists frames where MediaPipe detected something CVAT doesn't have
- **Issue:** Many false detections = thresholds too low or wrong model

## Troubleshooting

### 401 Unauthorized Error

**Problem:** Script can't authenticate with CVAT

**Solutions:**
1. Check if you're using correct credentials
2. Verify token is valid (not expired)
3. Check CVAT URL is correct
4. Try username/password instead of token

### No Frames Retrieved

**Problem:** Script can't download frames

**Solutions:**
1. Verify task/job ID is correct
2. Check frame numbers are within range
3. Verify authentication is working
4. Check CVAT server is accessible

### MediaPipe Service Not Responding

**Problem:** Can't connect to MediaPipe service

**Solutions:**
1. Verify MediaPipe service is running: `curl http://localhost:8000/health`
2. Check `--mediapipe-url` is correct
3. Check firewall/network settings

### Import Errors

**Problem:** Missing Python packages

**Solutions:**
```bash
cd serverless/mediapipe-service
pip install -r requirements.txt
```

## Tips

1. **Start Small:** Test with a few frames first (`--end-frame 10`)
2. **Use Step:** For large videos, use `--step 10` to sample every 10th frame
3. **Check Logs:** Script logs detailed information about each frame
4. **Visualize:** Use visualization images to understand coordinate errors
5. **Compare:** Run before and after fixes to measure improvement

## Next Steps

After running the debug script:

1. **Review Results:** Check `analysis_results.json` for statistics
2. **Examine Visualizations:** Look at comparison images to understand errors
3. **Identify Issues:** Use results to identify specific problems
4. **Apply Fixes:** Make changes to MediaPipe service based on findings
5. **Re-test:** Run script again to verify improvements

