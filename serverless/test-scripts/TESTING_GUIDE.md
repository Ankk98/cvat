# MediaPipe Testing Guide

This guide explains how to test the MediaPipe service for egocentric video annotation.

## Test Scripts

### 1. `test_mediapipe_ground_truth.py` (NEW - Recommended - CVAT-Independent)

**Purpose:** Validates MediaPipe predictions against ground truth annotations from local files.

**Features:**
- ✅ **CVAT-Independent:** Works with local annotation files (JSON/XML)
- Compares MediaPipe keypoints with ground truth annotations
- Calculates keypoint accuracy (distance errors)
- Validates hand detection rate
- Checks intersection with semantic segmentation masks (from SAM/Detectron2)
- Generates comprehensive accuracy reports

**Usage:**
```bash
cd serverless/test-scripts

# Basic test with ground truth annotations
python test_mediapipe_ground_truth.py \
    --dataset-path test-data/egocentric-hands \
    --annotations test-data/egocentric-hands/annotations.json \
    --mediapipe-url http://localhost:8000

# Test with semantic segmentation intersection
python test_mediapipe_ground_truth.py \
    --dataset-path test-data/egocentric-hands \
    --annotations test-data/egocentric-hands/annotations.json \
    --semantic-masks sam_results.json \
    --mediapipe-url http://localhost:8000
```

**Output:**
- JSON file with detailed comparison results
- Statistics: detection rate, keypoint accuracy, IoU metrics
- Per-image analysis

**See:** `README_GROUND_TRUTH_TESTING.md` for detailed guide

### 2. `download_gt_annotations.py` (NEW)

**Purpose:** Download or generate ground truth annotations for test datasets.

**Usage:**
```bash
# Create empty annotation structure
python download_gt_annotations.py \
    --dataset-path test-data/egocentric-hands \
    --format json \
    --output annotations.json

# Convert from existing annotations
python download_gt_annotations.py \
    --dataset-path test-data/egocentric-hands \
    --format json \
    --input existing_annotations.xml \
    --output annotations.json
```

### 3. `test_mediapipe_egocentric.py` (Updated)

**Purpose:** Tests MediaPipe on egocentric vision datasets with synthetic/real images.

**Features:**
- Tests on difficulty levels (easy/medium/hard)
- Analyzes pose and hand detection
- Performance metrics
- CVAT integration validation

**Usage:**
```bash
python test_mediapipe_egocentric.py \
    --dataset-path test-data/synthetic-egocentric \
    --difficulty easy \
    --max-samples 10
```

### 4. `debug_annotations.py` (MediaPipe Service - CVAT-Dependent)

**Purpose:** Debug script to compare CVAT annotations with MediaPipe service output.

**Usage:**
```bash
cd serverless/mediapipe-service

python debug_annotations.py \
    --task-id 14 \
    --job-id 10 \
    --username admin \
    --password password \
    --start-frame 0 \
    --end-frame 100
```

## Testing Workflow

### Option A: CVAT-Independent Testing (Recommended)

**Best for:** Testing against known ground truth annotations without CVAT dependency.

```bash
# 1. Download test images
python download_real_images.py --category egocentric-hands --count 20

# 2. Create annotation structure
python download_gt_annotations.py \
    --dataset-path test-data/egocentric-hands \
    --format json \
    --output test-data/egocentric-hands/annotations.json

# 3. (Manually add ground truth keypoints to annotations.json)

# 4. Run ground truth validation
python test_mediapipe_ground_truth.py \
    --dataset-path test-data/egocentric-hands \
    --annotations test-data/egocentric-hands/annotations.json
```

### Option B: CVAT-Dependent Testing

**Best for:** Testing against annotations already in CVAT.

```bash
# Use debug_annotations.py (see below)
```

### Step 1: Test MediaPipe Service Health

```bash
curl http://localhost:8000/health
```

Should return: `{"status": "healthy", "service": "mediapipe-pose"}`

### Step 2: Run Ground Truth Validation

This is the most comprehensive test - it validates against real annotations:

```bash
python test_mediapipe_ground_truth.py \
    --task-id 14 \
    --job-id 10 \
    --username admin \
    --password password \
    --start-frame 0 \
    --end-frame 50
```

### Step 3: Review Results

Check the generated JSON file in `test-results/`:
- `hand_detection_rate`: Should be >0.7 (70%)
- `keypoint_distance_errors`: Average should be <0.05 (5% of image size)
- `intersection_metrics`: IoU should be >0.3 for good intersection

### Step 4: Run Debug Script (if issues found)

```bash
cd serverless/mediapipe-service
python debug_annotations.py \
    --task-id 14 \
    --job-id 10 \
    --username admin \
    --password password
```

This generates comparison reports showing CVAT vs MediaPipe annotations.

## Expected Results

### Hand Detection Rate
- **Target:** >70% of frames with visible hands
- **Current:** ~50% (after fixes)

### Keypoint Accuracy
- **Target:** Average error <5% of image size
- **Good:** <10% of image size
- **Acceptable:** <15% of image size

### Semantic Segmentation Intersection
- **Target:** IoU >0.3 for hand regions
- **Good:** IoU >0.5

## Troubleshooting

### MediaPipe Service Not Responding

```bash
# Check if service is running
curl http://localhost:8000/health

# Check service logs
cd serverless/mediapipe-service
tail -f logs/mediapipe.log  # or check console output
```

### Authentication Errors

Use one of these methods:
1. Personal Access Token: `--token YOUR_TOKEN`
2. Username/Password: `--username admin --password password`
3. Environment variables: `export CVAT_USERNAME=admin CVAT_PASSWORD=password`

### Low Detection Rate

1. Check MediaPipe service logs for detection failures
2. Verify thresholds are set correctly (should be 0.3)
3. Check if images are being processed correctly
4. Verify hand detection is working (look for "Hands detected" in logs)

### Coordinate Errors

1. Run debug script to analyze errors
2. Check if image resolution matches between CVAT and MediaPipe
3. Verify coordinate transformation is correct
4. Check for image preprocessing issues

## Test Data

### Using CVAT Tasks/Jobs

The ground truth test uses actual CVAT annotations:
- Requires a CVAT task/job with skeleton annotations
- Should have hand keypoints annotated
- Optional: semantic segmentation masks for intersection testing

### Using Test Images

The egocentric test uses test images:
- `test-data/synthetic-egocentric/`: Synthetic test images
- `test-data/egocentric-hands/`: Real hand images
- `test-data/egocentric-kitchen/`: Kitchen scene images

## Continuous Testing

### Run All Tests

```bash
cd serverless/test-scripts
./run_all_tests.sh
```

### Automated Validation

Add to CI/CD pipeline:
```bash
# Test MediaPipe service
python test_mediapipe_ground_truth.py \
    --task-id $CVAT_TASK_ID \
    --token $CVAT_TOKEN \
    --start-frame 0 \
    --end-frame 100

# Check if metrics meet thresholds
python -c "
import json
with open('test-results/latest.json') as f:
    data = json.load(f)
    stats = data['statistics']
    assert stats['hand_detection_rate'] > 0.7, 'Hand detection rate too low'
    assert np.mean(stats['keypoint_distance_errors']) < 0.05, 'Keypoint accuracy too low'
"
```

## Metrics Explained

### Hand Detection Rate
Percentage of frames with ground truth hands where MediaPipe also detected hands.

### Keypoint Distance Error
Normalized distance between ground truth and MediaPipe keypoints:
- `normalized_distance = distance / max(image_width, image_height)`
- Lower is better

### Intersection over Union (IoU)
For semantic segmentation intersection:
- `IoU = intersection_area / union_area`
- Measures overlap between hand keypoint region and semantic mask
- Higher is better (1.0 = perfect overlap)

## Next Steps

1. **Run ground truth validation** on your egocentric video
2. **Review results** and identify issues
3. **Adjust thresholds** if needed (in `app.py`)
4. **Re-test** to verify improvements
5. **Document** any issues or improvements needed

