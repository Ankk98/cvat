# MediaPipe Service Testing and Fixes Summary

## Issues Fixed

### 1. Hand Landmark Visibility Bug ✅

**Problem:** Hand landmarks don't have `visibility` or `presence` attributes (they're `None`), causing comparison errors:
```
ERROR: '>' not supported between instances of 'NoneType' and 'float'
```

**Fix Applied:**
- Updated `prioritize_hands()` function to handle `None` visibility/presence
- Updated `process_combined_results()` to safely extract confidence from hand landmarks
- Added fallback to `1.0` (visible) when attributes are missing

**Files Changed:**
- `serverless/mediapipe-service/app.py` (lines 150-156, 298-305)

### 2. API Endpoint Format ✅

**Problem:** CVAT API requires `type=frame` parameter, which was missing.

**Fix Applied:**
- Added `type=frame` parameter to frame retrieval API calls
- Updated both task and job endpoints

**Files Changed:**
- `serverless/mediapipe-service/debug_annotations.py` (line 124)

## New Test Scripts Created

### 1. `test_mediapipe_ground_truth.py` ✅

**Purpose:** Comprehensive validation against ground truth annotations from CVAT.

**Features:**
- ✅ Loads ground truth skeleton annotations from CVAT
- ✅ Compares MediaPipe predictions with ground truth keypoints
- ✅ Calculates keypoint accuracy (distance errors, normalized)
- ✅ Validates hand detection rate
- ✅ Checks intersection with semantic segmentation masks (IoU)
- ✅ Generates detailed accuracy reports

**Key Metrics:**
- Hand detection rate: % of frames with GT hands where MediaPipe detected hands
- Keypoint accuracy: Average distance error (normalized by image size)
- Semantic intersection: IoU between hand keypoints and segmentation masks

**Usage:**
```bash
cd serverless/test-scripts

python test_mediapipe_ground_truth.py \
    --cvat-url http://localhost:8080 \
    --task-id 14 \
    --job-id 10 \
    --username admin \
    --password password \
    --start-frame 0 \
    --end-frame 100
```

### 2. Updated `debug_annotations.py` ✅

**Improvements:**
- ✅ Added authentication support (PAT, username/password, env vars)
- ✅ Fixed API endpoint format (`type=frame` parameter)
- ✅ Better error logging
- ✅ Comprehensive comparison analysis

## Test Results Analysis

Based on your test run:

```
Total frames analyzed: 101
Frames with CVAT annotations: 30
Frames with MediaPipe detections: 51
Frames with both: 26
```

**Observations:**
- ✅ MediaPipe is detecting more frames than have ground truth (51 vs 30)
- ✅ 26 frames have both GT and MediaPipe annotations (good for validation)
- ⚠️ 30 frames have GT but no MediaPipe detection (need investigation)
- ⚠️ 25 frames have MediaPipe but no GT (may be false positives or missing GT)

**Next Steps:**
1. Run `test_mediapipe_ground_truth.py` to get detailed accuracy metrics
2. Check frames with GT but no MediaPipe detection
3. Verify frames with MediaPipe but no GT (may need manual review)

## Testing Workflow

### Quick Test
```bash
# 1. Test service health
curl http://localhost:8000/health

# 2. Run ground truth validation
cd serverless/test-scripts
python test_mediapipe_ground_truth.py \
    --task-id 14 --job-id 10 \
    --username admin --password password \
    --start-frame 0 --end-frame 50

# 3. Review results
cat test-results/mediapipe_ground_truth_test_*.json | jq '.statistics'
```

### Comprehensive Test
```bash
# Full validation with all metrics
python test_mediapipe_ground_truth.py \
    --task-id 14 --job-id 10 \
    --username admin --password password \
    --start-frame 0 --end-frame 100 \
    --output-dir comprehensive_results
```

## Expected Metrics

### Hand Detection Rate
- **Target:** >70%
- **Current:** ~50% (needs improvement)
- **Action:** May need to lower thresholds further or improve hand detection

### Keypoint Accuracy
- **Target:** Average error <5% of image size
- **Good:** <10% of image size
- **Action:** Verify coordinate transformation is correct

### Semantic Segmentation Intersection
- **Target:** IoU >0.3 for hand regions
- **Good:** IoU >0.5
- **Action:** Ensure hand keypoints form proper bounding regions

## Files Created/Updated

### New Files
1. `serverless/test-scripts/test_mediapipe_ground_truth.py` - Ground truth validation test
2. `serverless/test-scripts/TESTING_GUIDE.md` - Comprehensive testing guide
3. `serverless/mediapipe-service/TESTING_AND_FIXES_SUMMARY.md` - This file

### Updated Files
1. `serverless/mediapipe-service/app.py` - Fixed hand landmark visibility bug
2. `serverless/mediapipe-service/debug_annotations.py` - Added authentication, fixed API endpoint

## Next Steps

1. **Run Ground Truth Test:**
   ```bash
   python test_mediapipe_ground_truth.py --task-id 14 --job-id 10 --username admin --password password
   ```

2. **Review Results:**
   - Check hand detection rate
   - Review keypoint accuracy
   - Analyze semantic segmentation intersection

3. **Iterate on Fixes:**
   - Adjust thresholds if detection rate is low
   - Fix coordinate transformation if errors are high
   - Improve hand prioritization if needed

4. **Document Findings:**
   - Update test results
   - Document any issues found
   - Create improvement plan if needed

## Troubleshooting

### If Hand Detection Rate is Low
- Check MediaPipe service logs for detection failures
- Verify thresholds are 0.3 (not higher)
- Check if hands are actually visible in frames
- Consider lowering thresholds further (to 0.2)

### If Keypoint Accuracy is Poor
- Run debug script to visualize errors
- Check coordinate transformation
- Verify image resolution matches
- Check for image preprocessing issues

### If Semantic Intersection is Low
- Verify semantic masks are correctly loaded
- Check hand keypoint bounding regions
- Ensure masks and keypoints are in same coordinate system

## References

- **Testing Guide:** `serverless/test-scripts/TESTING_GUIDE.md`
- **Fix Plan:** `serverless/mediapipe-service/MEDIAPIPE_EGOCENTRIC_FIX_PLAN.md`
- **Fixes Applied:** `serverless/mediapipe-service/FIXES_APPLIED.md`
- **Debug Script:** `serverless/mediapipe-service/debug_annotations.py`

