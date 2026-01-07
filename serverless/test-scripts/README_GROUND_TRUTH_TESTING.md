# MediaPipe Ground Truth Testing Guide

## Overview

This guide explains how to test MediaPipe hand pose estimation against ground truth annotations and validate intersection with semantic segmentation.

## Quick Start

### 1. Prepare Test Data

```bash
cd serverless/test-scripts

# Download test images (if not already done)
python download_real_images.py --category egocentric-hands --count 10

# Create ground truth annotations structure
python download_gt_annotations.py \
    --dataset-path test-data/egocentric-hands \
    --format json \
    --output test-data/egocentric-hands/annotations.json
```

### 2. Run Ground Truth Test

```bash
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

## Ground Truth Annotation Format

### JSON Format

```json
{
  "001.jpg": {
    "skeletons": [
      {
        "label": "person",
        "elements": [
          {
            "label": "left_wrist",
            "points": [100.5, 200.3],
            "outside": false,
            "occluded": false,
            "attributes": [
              {"name": "confidence", "value": "1.0"}
            ]
          },
          {
            "label": "right_wrist",
            "points": [300.2, 250.1],
            "outside": false,
            "occluded": false
          }
        ]
      }
    ],
    "masks": [
      {
        "label": "hand",
        "rle": [0, 100, 200, 50, ...],
        "left": 50,
        "top": 150,
        "width": 100,
        "height": 100
      }
    ]
  }
}
```

### CVAT XML Format

```xml
<annotations>
  <version>1.1</version>
  <image name="001.jpg" id="0">
    <skeleton label="person">
      <points label="left_wrist" points="100.5,200.3" outside="0" occluded="0"/>
      <points label="right_wrist" points="300.2,250.1" outside="0" occluded="0"/>
    </skeleton>
    <mask label="hand" rle="0,100,200,50,..." left="50" top="150" width="100" height="100"/>
  </image>
</annotations>
```

## Semantic Segmentation Masks Format

### SAM/Detectron2 Results JSON

```json
{
  "001.jpg": [
    {
      "rle": [0, 100, 200, 50, ...],
      "label": "hand",
      "score": 0.95
    }
  ],
  "002.jpg": [
    {
      "polygon": [100, 150, 200, 150, 200, 250, 100, 250],
      "label": "object",
      "score": 0.87
    }
  ]
}
```

## Workflow

### Step 1: Download Images

```bash
python download_real_images.py \
    --category egocentric-hands \
    --count 20 \
    --output-dir test-data/egocentric-hands
```

### Step 2: Create/Download Ground Truth Annotations

**Option A: Create empty structure**
```bash
python download_gt_annotations.py \
    --dataset-path test-data/egocentric-hands \
    --format json \
    --output test-data/egocentric-hands/annotations.json
```

**Option B: Load from existing file**
```bash
python download_gt_annotations.py \
    --dataset-path test-data/egocentric-hands \
    --format json \
    --input existing_annotations.xml \
    --output test-data/egocentric-hands/annotations.json
```

### Step 3: Generate Semantic Segmentation Masks (Optional)

Run SAM or Detectron2 on images to generate semantic masks:

```bash
# Using SAM
python test_sam_auto_egocentric.py \
    --dataset-path test-data/egocentric-hands \
    --output-dir sam_results

# Using Detectron2
python test_detectron2_egocentric.py \
    --dataset-path test-data/egocentric-hands \
    --output-dir detectron2_results
```

### Step 4: Run Ground Truth Validation

```bash
python test_mediapipe_ground_truth.py \
    --dataset-path test-data/egocentric-hands \
    --annotations test-data/egocentric-hands/annotations.json \
    --semantic-masks sam_results/results.json \
    --mediapipe-url http://localhost:8000 \
    --output-dir test-results
```

## Expected Output

### Console Summary

```
================================================================================
GROUND TRUTH VALIDATION SUMMARY
================================================================================
Total images analyzed: 20
Images with ground truth: 15
Images with MediaPipe detections: 18
Images with both: 12
Hand detection rate: 80.00%

Keypoint Accuracy:
  Average error: 3.45% of image size
  Median error: 2.10% of image size
  95th percentile error: 8.20% of image size

Semantic Segmentation Intersection:
  Average IoU: 0.452
================================================================================
```

### JSON Results File

Results are saved to `test-results/mediapipe_ground_truth_test_<timestamp>.json`:

```json
{
  "statistics": {
    "total_images": 20,
    "images_with_gt": 15,
    "images_with_mp": 18,
    "images_with_both": 12,
    "hand_detection_rate": 0.8,
    "keypoint_distance_errors": [0.023, 0.045, ...],
    "intersection_metrics": [
      {"iou": 0.45, "intersection_area": 1234, "union_area": 2734}
    ]
  },
  "detailed_results": [
    {
      "image": "001.jpg",
      "gt_keypoints": {...},
      "mp_keypoints": {...},
      "matches": [...],
      "keypoint_errors": [...],
      "intersection_metrics": {...}
    }
  ]
}
```

## Metrics Explained

### Hand Detection Rate
Percentage of images with ground truth hands where MediaPipe also detected hands.
- **Target:** >70%
- **Good:** >80%
- **Excellent:** >90%

### Keypoint Accuracy
Normalized distance error between ground truth and MediaPipe keypoints.
- **Target:** Average <5% of image size
- **Good:** Average <3% of image size
- **Excellent:** Average <2% of image size

### Semantic Segmentation Intersection (IoU)
Intersection over Union between hand keypoint regions and semantic masks.
- **Target:** IoU >0.3
- **Good:** IoU >0.5
- **Excellent:** IoU >0.7

## Integration with Other Models

### Using SAM Results

```bash
# 1. Run SAM on images
python test_sam_auto_egocentric.py \
    --dataset-path test-data/egocentric-hands \
    --output-dir sam_results

# 2. Convert SAM results to semantic masks format
# (SAM results should include mask data)

# 3. Run ground truth test with SAM masks
python test_mediapipe_ground_truth.py \
    --dataset-path test-data/egocentric-hands \
    --annotations annotations.json \
    --semantic-masks sam_results/results.json
```

### Using Detectron2 Results

```bash
# 1. Run Detectron2 on images
python test_detectron2_egocentric.py \
    --dataset-path test-data/egocentric-hands \
    --output-dir detectron2_results

# 2. Run ground truth test with Detectron2 masks
python test_mediapipe_ground_truth.py \
    --dataset-path test-data/egocentric-hands \
    --annotations annotations.json \
    --semantic-masks detectron2_results/results.json
```

## Troubleshooting

### No Ground Truth Annotations Found

**Problem:** Test reports 0 images with ground truth

**Solutions:**
1. Check annotation file path is correct
2. Verify annotation file format (JSON or XML)
3. Ensure image names in annotations match image filenames
4. Check annotation file structure matches expected format

### No Semantic Masks Found

**Problem:** Intersection metrics are empty

**Solutions:**
1. Verify semantic masks file path
2. Check semantic masks format matches expected structure
3. Ensure image names in masks match image filenames
4. Verify masks contain valid RLE or polygon data

### MediaPipe Service Not Responding

**Problem:** Cannot connect to MediaPipe service

**Solutions:**
1. Check service is running: `curl http://localhost:8000/health`
2. Verify `--mediapipe-url` is correct
3. Check firewall/network settings

### Low Detection Rate

**Problem:** Hand detection rate <50%

**Solutions:**
1. Check MediaPipe service logs
2. Verify thresholds are set correctly (should be 0.3)
3. Check if images actually contain visible hands
4. Consider lowering thresholds further

## Example Workflow

```bash
# Complete workflow example
cd serverless/test-scripts

# 1. Download images
python download_real_images.py --category egocentric-hands --count 20

# 2. Create annotation structure
python download_gt_annotations.py \
    --dataset-path test-data/egocentric-hands \
    --format json \
    --output test-data/egocentric-hands/annotations.json

# 3. (Manually edit annotations.json to add ground truth keypoints)

# 4. Run SAM for semantic masks
python test_sam_auto_egocentric.py \
    --dataset-path test-data/egocentric-hands \
    --output-dir sam_results

# 5. Run ground truth validation
python test_mediapipe_ground_truth.py \
    --dataset-path test-data/egocentric-hands \
    --annotations test-data/egocentric-hands/annotations.json \
    --semantic-masks sam_results/results.json

# 6. Review results
cat test-results/mediapipe_ground_truth_test_*.json | jq '.statistics'
```

## Next Steps

1. **Create Ground Truth:** Manually annotate test images or download from public datasets
2. **Run Tests:** Execute ground truth validation
3. **Analyze Results:** Review accuracy metrics
4. **Iterate:** Adjust MediaPipe thresholds or improve detection logic
5. **Document:** Record findings and improvements

