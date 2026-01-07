# MediaPipe Egocentric Video Fixes Applied

## Summary

This document summarizes the fixes applied to improve MediaPipe auto-annotation for egocentric videos in CVAT.

## Issues Fixed

### 1. Low Detection Thresholds ✅

**Problem:** Detection thresholds were too high (0.5) for egocentric videos with challenging lighting/angles.

**Fix Applied:**
- Lowered `min_pose_detection_confidence` from 0.5 to 0.3
- Lowered `min_pose_presence_confidence` from 0.5 to 0.3
- Lowered `min_hand_detection_confidence` from 0.5 to 0.3
- Lowered `min_hand_presence_confidence` from 0.5 to 0.3
- Lowered `min_tracking_confidence` from 0.5 to 0.3

**Expected Impact:** Detection rate should increase from ~10% to ~30-40%

### 2. Hand Detection Prioritization ✅

**Problem:** Hands were not being detected reliably in egocentric videos. Pose detection was prioritized over hand detection.

**Fix Applied:**
- Changed detection order: hands detection now runs first
- Added `prioritize_hands()` function that scores hands based on:
  - Distance from center of frame (50% weight)
  - Number of visible keypoints (30% weight)
  - Average confidence (20% weight)
- Hands closer to center are prioritized (important for egocentric videos)

**Expected Impact:** Better hand detection, especially for hands in center of frame

### 3. Relaxed Filtering Logic ✅

**Problem:** Filtering logic was too strict, requiring at least 2 visible keypoints for any detection.

**Fix Applied:**
- If hands detected: require at least 3 hand keypoints (instead of 2 total)
- If only pose detected: require at least 5 pose keypoints
- This allows hand-only detections to pass through

**Expected Impact:** More frames annotated, especially frames with only hands visible

### 4. Hand Prioritization for Multiple Hands ✅

**Problem:** When multiple hands detected, all were processed equally without considering position.

**Fix Applied:**
- Added `prioritize_hands()` function that selects up to 2 hands closest to center
- Hands are scored and sorted by proximity to center, visibility, and confidence
- Only top 2 hands are processed (reduces noise from background hands)

**Expected Impact:** Better focus on hands in center of frame (most relevant for egocentric videos)

### 5. Coordinate Transformation Fixes ✅

**Problem:** Hand pose annotations appeared at bottom-right corner instead of actual hand locations.

**Root Causes Identified:**
- MediaPipe sometimes returns normalized coordinates outside valid [0,1] range
- Image format mismatch: MediaPipe expects RGB but received BGR from OpenCV
- Coordinate clamping was missing

**Fixes Applied:**
- **RGB Format Conversion:** Added `cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)` before creating MediaPipe image
- **Coordinate Clamping:** Clamp normalized coordinates to [0.0, 1.0] range before pixel conversion
- **Logging:** Added diagnostic logging for image dimensions and coordinate ranges

**Code Changes:**
```python
# Convert BGR to RGB (MediaPipe expects RGB)
cv2_rgb = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2_rgb)

# Clamp coordinates to valid range
x_norm = max(0.0, min(1.0, landmark.x))
y_norm = max(0.0, min(1.0, landmark.y))
x_pixel = x_norm * image_width
y_pixel = y_norm * image_height
```

**Expected Impact:** Annotations now appear at correct hand locations instead of bottom-right corner

## Code Changes

### Modified Files

1. **`serverless/mediapipe-service/app.py`**
   - Lowered detection thresholds (lines 77-79, 107-109)
   - Added `prioritize_hands()` function (lines 120-170)
   - Modified `process_combined_results()` to use prioritized hands (lines 258-260)
   - Changed detection order: hands first, then pose (lines 339-340)
   - Relaxed filtering logic (lines 330-345)

### New Files

1. **`serverless/mediapipe-service/debug_annotations.py`**
   - Debug script to compare CVAT annotations with MediaPipe output
   - Analyzes detection rate, coordinate accuracy, and errors
   - Generates visualization comparing annotations

2. **`serverless/mediapipe-service/MEDIAPIPE_EGOCENTRIC_FIX_PLAN.md`**
   - Comprehensive analysis of issues
   - Detailed fix plan with phases
   - Expected outcomes and success criteria

## Testing

### Manual Testing

1. **Restart MediaPipe Service:**
   ```bash
   cd serverless/mediapipe-service
   ./stop.sh
   ./start.sh
   ```

2. **Run Auto-Annotation in CVAT:**
   - Open task/job in CVAT
   - Select "MediaPipe Pose + Hands" from auto-annotation dropdown
   - Run annotation on egocentric video
   - Check detection rate (should be >30% instead of ~10%)

3. **Verify Hand Detection:**
   - Check frames with visible hands
   - Verify hand keypoints are detected
   - Verify keypoints are in correct positions

### Automated Testing

Use the debug script to analyze annotations:

```bash
cd serverless/mediapipe-service
python debug_annotations.py \
    --cvat-url http://localhost:8080 \
    --task-id 14 \
    --job-id 10 \
    --start-frame 0 \
    --end-frame 100 \
    --output-dir debug_results
```

This will:
- Download frames from CVAT
- Send to MediaPipe service
- Compare with CVAT annotations
- Generate comparison visualizations
- Report detection rate and accuracy metrics

## Expected Improvements

### Detection Rate
- **Before:** ~10% of frames annotated
- **After:** >30-40% of frames annotated (target: >60% with further tuning)

### Hand Detection
- **Before:** Hands rarely detected
- **After:** Hands detected in >50% of frames with visible hands (target: >70%)

### Accuracy
- **Before:** Many incorrect keypoint positions
- **After:** Improved accuracy (verify with debug script)

### Hand Prioritization
- **Before:** All hands processed equally
- **After:** Hands in center prioritized (better for egocentric videos)

## Next Steps

### Immediate
1. Test fixes with egocentric video
2. Monitor detection rate improvement
3. Verify coordinate accuracy with debug script

### Short-term (if needed)
1. Further lower thresholds if detection rate still low
2. Adjust hand prioritization weights based on results
3. Fix coordinate transformation if errors found

### Long-term
1. Add egocentric mode flag for configurable behavior
2. Implement image preprocessing (contrast enhancement)
3. Consider using different MediaPipe models (full vs lite)

## Troubleshooting

### If Detection Rate Still Low

1. **Check MediaPipe Service Logs:**
   ```bash
   cd serverless/mediapipe-service
   tail -f logs/mediapipe.log  # or check console output
   ```

2. **Verify Thresholds:**
   - Check that thresholds are actually 0.3 (not 0.5)
   - May need to lower further to 0.2 for very challenging videos

3. **Check Hand Detection:**
   - Look for "Hands detected: []" in logs
   - If hands not detected, may need to lower hand thresholds further

### If Coordinates Still Wrong

1. **Run Debug Script:**
   ```bash
   python debug_annotations.py --task-id 14 --job-id 10
   ```

2. **Check Coordinate Errors:**
   - Review `debug_results/analysis_results.json`
   - Look for frames with high coordinate errors
   - Verify image resolution matches between CVAT and MediaPipe

3. **Verify Coordinate Transformation:**
   - MediaPipe returns normalized coordinates (0-1)
   - Conversion: `x_pixel = x_normalized * image_width`
   - Check if image is being resized before processing

### If False Body Parts Detected

1. **Check Pose Detection:**
   - Look for pose detections when only hands visible
   - May need to disable pose detection for hand-only frames
   - Or increase pose detection threshold further

2. **Review Filtering Logic:**
   - Check if pose keypoints are being filtered correctly
   - May need to add logic to filter low-confidence body parts

## Monitoring

### Key Metrics to Track

1. **Detection Rate:** Percentage of frames with annotations
2. **Hand Detection Rate:** Percentage of frames with hand keypoints
3. **Coordinate Accuracy:** Average distance error (from debug script)
4. **False Positive Rate:** Incorrect body parts detected

### Logging

MediaPipe service logs include:
- Detection results (pose/hands detected)
- Number of visible keypoints
- Filtering decisions (insufficient keypoints)

Monitor logs to understand detection patterns and identify issues.

## Rollback

If fixes cause issues, rollback by:

1. **Restore Original Thresholds:**
   ```python
   # In app.py, change thresholds back to 0.5
   min_pose_detection_confidence=0.5,
   min_pose_presence_confidence=0.5,
   min_hand_detection_confidence=0.5,
   min_hand_presence_confidence=0.5,
   ```

2. **Remove Hand Prioritization:**
   - Comment out `prioritize_hands()` call
   - Use all detected hands instead

3. **Restore Original Filtering:**
   - Change back to requiring 2 visible keypoints minimum

## References

- **Fix Plan:** `MEDIAPIPE_EGOCENTRIC_FIX_PLAN.md`
- **Debug Script:** `debug_annotations.py`
- **MediaPipe Docs:** https://developers.google.com/mediapipe
- **CVAT Documentation:** Main CVAT docs

