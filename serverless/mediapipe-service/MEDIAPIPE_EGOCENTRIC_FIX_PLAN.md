# MediaPipe Egocentric Video Annotation Fix Plan

## Executive Summary

This document outlines the issues identified with MediaPipe auto-annotation for egocentric videos and provides a comprehensive plan to fix them. The main problems are:

1. **Low detection rate**: Only ~10% of frames are being annotated
2. **Incorrect keypoint locations**: Annotations appear in wrong positions
3. **Wrong keypoint types**: Body parts detected when only hands are visible
4. **Hand detection failure**: MediaPipe Hands detector not detecting hands in egocentric videos
5. **No hand prioritization**: When multiple hands present, should focus on center of frame

## Issues Identified

### 1. Hand Detection Not Working

**Symptoms:**
- Logs show: `"Hands detected: []"` for most frames
- Even when hands are clearly visible in frames
- MediaPipe Hands detector returns empty results

**Root Causes:**
- MediaPipe Hands detector has high confidence thresholds (0.5)
- Egocentric videos have challenging lighting/angles
- Hands may be partially occluded or at unusual angles
- Detector optimized for full-body poses, not isolated hands

**Evidence from Logs:**
```
INFO:__main__:Pose detected: [], Hands detected: []
INFO:__main__:Insufficient visible keypoints detected (0), skipping
```

### 2. Pose Detection Issues

**Symptoms:**
- Pose detector detects body parts when only hands visible
- Keypoints appear in wrong locations
- Annotations show body parts (shoulders, hips, knees) that aren't visible

**Root Causes:**
- MediaPipe Pose detector is designed for full-body poses
- In egocentric videos, only hands/arms are typically visible
- Detector may be hallucinating body parts based on partial arm detection
- Coordinate transformation may be incorrect

**Evidence:**
- Images show only hands, but annotations include shoulders, hips, knees
- Keypoint positions don't align with actual hand positions

### 3. Low Detection Rate (~10%)

**Symptoms:**
- Most frames return no detections
- Only occasional frames get annotated
- Many frames with visible hands return empty results

**Root Causes:**
- High confidence thresholds filtering out valid detections
- Minimum keypoint requirement (2 visible keypoints) too strict
- Hand detection failure means pose-only detections get filtered
- Egocentric video characteristics (fisheye distortion, unusual angles)

### 4. Coordinate Transformation Issues

**Symptoms:**
- Keypoints appear in wrong positions
- Annotations don't align with actual hand positions
- Possible resolution mismatch

**Potential Root Causes:**
- MediaPipe returns normalized coordinates (0-1)
- Conversion to pixel coordinates may be incorrect
- Image resolution changes between CVAT and MediaPipe
- Coordinate system mismatch (MediaPipe uses different origin)

**Code Location:**
```python
# app.py line 195-196
"points": [
    landmark.x * image_width,
    landmark.y * image_height
]
```

### 5. No Hand Prioritization

**Symptoms:**
- When multiple hands detected, no prioritization
- Should focus on hands in center of frame for egocentric videos

**Root Causes:**
- Current implementation processes all detected hands equally
- No spatial filtering or prioritization logic

## Detailed Analysis

### MediaPipe Service Code Issues

#### 1. Detection Thresholds Too High

**Location:** `app.py` lines 77-79, 107-109

```python
min_pose_detection_confidence=0.5,
min_pose_presence_confidence=0.5,
min_tracking_confidence=0.5,
```

**Problem:** These thresholds are too high for egocentric videos where:
- Lighting may be suboptimal
- Angles are unusual
- Hands may be partially occluded

**Fix:** Lower thresholds to 0.3 or make configurable

#### 2. Hand Detection Not Prioritized

**Location:** `app.py` lines 339-340

```python
# Run pose detection
pose_results = pose_detector.detect(mp_image)

# Run hands detection
hands_results = hands_detector.detect(mp_image)
```

**Problem:** Both detectors run independently. For egocentric videos, hands should be prioritized.

**Fix:**
- Run hands detection first
- If hands detected, prioritize hand keypoints
- Only use pose detection if no hands found

#### 3. Filtering Logic Too Strict

**Location:** `app.py` lines 260-263

```python
# Require at least 2 visible keypoints for a valid skeleton
if len(visible_keypoints) < 2:
    logger.info(f"Insufficient visible keypoints detected ({len(visible_keypoints)}), skipping")
    return []
```

**Problem:** For egocentric videos with only hands visible, this may filter out valid single-hand detections.

**Fix:**
- Lower threshold to 1 for hand-only detections
- Or require at least 3 hand keypoints if hands detected

#### 4. Coordinate System Verification Needed

**Location:** `app.py` lines 195-196

**Issue:** Need to verify MediaPipe coordinate system matches CVAT expectations.

**MediaPipe:** Returns normalized coordinates (0-1) where:
- (0, 0) = top-left
- (1, 1) = bottom-right

**CVAT:** Expects pixel coordinates where:
- (0, 0) = top-left
- (width, height) = bottom-right

**Verification Needed:** Test with known hand positions to verify coordinate transformation.

#### 5. No Hand Prioritization Logic

**Location:** `app.py` lines 216-245

**Problem:** When multiple hands detected, processes all equally. For egocentric videos, should prioritize:
- Hands closer to center of frame
- Hands with more visible keypoints
- Hands with higher confidence

**Fix:** Add hand prioritization function

## Fix Plan

### Phase 1: Immediate Fixes (High Priority)

#### 1.1 Lower Detection Thresholds

**File:** `serverless/mediapipe-service/app.py`

**Changes:**
- Lower `min_pose_detection_confidence` to 0.3
- Lower `min_pose_presence_confidence` to 0.3
- Lower `min_hand_detection_confidence` to 0.3
- Lower `min_hand_presence_confidence` to 0.3

**Expected Impact:** Increase detection rate from ~10% to ~30-40%

#### 1.2 Prioritize Hand Detection

**File:** `serverless/mediapipe-service/app.py`

**Changes:**
- Run hands detection first
- If hands detected, create skeleton from hands only
- Only add pose keypoints if they complement hand detection
- For egocentric videos, prefer hand-only skeletons over pose-only

**Expected Impact:** Better hand detection, fewer false body part detections

#### 1.3 Relax Filtering Logic

**File:** `serverless/mediapipe-service/app.py`

**Changes:**
- If hands detected, require minimum 3 hand keypoints (instead of 2 total)
- If only pose detected, require minimum 5 pose keypoints
- Lower threshold for hand-only detections

**Expected Impact:** More frames annotated, especially hand-only frames

### Phase 2: Hand Prioritization (Medium Priority)

#### 2.1 Add Hand Prioritization Function

**File:** `serverless/mediapipe-service/app.py`

**New Function:**
```python
def prioritize_hands(hands_results, image_width, image_height, center_weight=0.5):
    """
    Prioritize hands based on:
    - Distance from center of frame
    - Number of visible keypoints
    - Average confidence
    """
    # Calculate center of frame
    center_x, center_y = image_width / 2, image_height / 2

    # Score each hand
    scored_hands = []
    for hand_idx, hand_landmarks in enumerate(hands_results.hand_landmarks):
        # Calculate distance from center
        wrist = hand_landmarks[0]  # Wrist is first keypoint
        dist_from_center = np.sqrt(
            (wrist.x * image_width - center_x)**2 +
            (wrist.y * image_height - center_y)**2
        )
        normalized_dist = dist_from_center / max(image_width, image_height)

        # Count visible keypoints
        visible_count = sum(1 for lm in hand_landmarks
                          if getattr(lm, 'visibility', 1.0) > 0.3)

        # Average confidence
        avg_confidence = np.mean([getattr(lm, 'visibility', 1.0)
                                 for lm in hand_landmarks])

        # Combined score (lower distance = higher score)
        score = (1 - normalized_dist) * center_weight + \
                (visible_count / 21) * 0.3 + \
                avg_confidence * 0.2

        scored_hands.append((score, hand_idx, hand_landmarks))

    # Sort by score (highest first)
    scored_hands.sort(reverse=True, key=lambda x: x[0])

    # Return top 2 hands (or all if less than 2)
    return [hand for _, _, hand in scored_hands[:2]]
```

**Expected Impact:** Better hand selection for egocentric videos

### Phase 3: Coordinate Verification and Fix (High Priority)

#### 3.1 Add Coordinate Verification

**File:** `serverless/mediapipe-service/debug_annotations.py`

**New Function:** Use debug script to verify coordinates

**Steps:**
1. Create test images with known hand positions
2. Send to MediaPipe service
3. Compare returned coordinates with expected positions
4. Verify coordinate transformation is correct

#### 3.2 Fix Coordinate Transformation if Needed

**File:** `serverless/mediapipe-service/app.py`

**Potential Fixes:**
- Verify MediaPipe coordinate system (may need to flip Y axis)
- Check if image is being resized before processing
- Ensure image dimensions match between CVAT and MediaPipe

### Phase 4: Egocentric-Specific Optimizations (Medium Priority)

#### 4.1 Add Egocentric Mode

**File:** `serverless/mediapipe-service/app.py`

**New Parameter:** `egocentric_mode` (default: False)

**When enabled:**
- Prioritize hands over pose
- Lower thresholds further
- Focus on center of frame
- Filter out low-confidence body parts

#### 4.2 Improve Hand Detection for Egocentric Videos

**Changes:**
- Use MediaPipe Hands detector with lower thresholds
- Add image preprocessing (contrast enhancement, noise reduction)
- Consider using different MediaPipe model (full vs lite)

### Phase 5: Testing and Validation

#### 5.1 Create Test Suite

**File:** `serverless/mediapipe-service/test_egocentric.py`

**Tests:**
- Test with real egocentric video frames
- Verify detection rate improvement
- Verify coordinate accuracy
- Test hand prioritization

#### 5.2 Use Debug Script

**File:** `serverless/mediapipe-service/debug_annotations.py`

**Usage:**
```bash
python debug_annotations.py \
    --cvat-url http://localhost:8080 \
    --task-id 14 \
    --job-id 10 \
    --start-frame 0 \
    --end-frame 100 \
    --output-dir debug_results
```

## Implementation Steps

### Step 1: Immediate Fixes (1-2 hours)

1. Lower detection thresholds in `app.py`
2. Modify detection logic to prioritize hands
3. Relax filtering logic
4. Test with sample frames

### Step 2: Hand Prioritization (1-2 hours)

1. Implement `prioritize_hands()` function
2. Integrate into `process_combined_results()`
3. Test with multiple hands in frame

### Step 3: Coordinate Verification (2-3 hours)

1. Run debug script on problematic frames
2. Analyze coordinate errors
3. Fix coordinate transformation if needed
4. Verify fixes with test images

### Step 4: Testing (2-3 hours)

1. Test with full video sequence
2. Measure detection rate improvement
3. Verify coordinate accuracy
4. Document results

## Expected Outcomes

### Detection Rate
- **Current:** ~10% of frames annotated
- **Target:** >60% of frames annotated
- **Method:** Lower thresholds, prioritize hands, relax filtering

### Accuracy
- **Current:** Many incorrect keypoint positions
- **Target:** >90% of keypoints within 5% of image size
- **Method:** Fix coordinate transformation, verify with debug script

### Hand Detection
- **Current:** Hands rarely detected
- **Target:** Hands detected in >70% of frames with visible hands
- **Method:** Prioritize hand detection, lower thresholds, improve filtering

## Monitoring and Validation

### Metrics to Track

1. **Detection Rate:** Percentage of frames with annotations
2. **Hand Detection Rate:** Percentage of frames with hand keypoints
3. **Coordinate Accuracy:** Average distance error between CVAT and MediaPipe
4. **False Positive Rate:** Percentage of frames with incorrect body parts

### Validation Process

1. Run debug script on sample frames
2. Compare MediaPipe output with manual annotations
3. Measure improvement metrics
4. Iterate on fixes based on results

## Risk Mitigation

### Risks

1. **Lowering thresholds too much:** May increase false positives
   - **Mitigation:** Start with moderate reduction, test, adjust

2. **Breaking existing functionality:** Changes may affect other use cases
   - **Mitigation:** Add egocentric mode flag, make configurable

3. **Coordinate transformation errors:** May introduce new bugs
   - **Mitigation:** Thorough testing with debug script, verify with known positions

## Timeline

- **Phase 1 (Immediate Fixes):** 1-2 hours
- **Phase 2 (Hand Prioritization):** 1-2 hours
- **Phase 3 (Coordinate Verification):** 2-3 hours
- **Phase 4 (Optimizations):** 2-3 hours
- **Phase 5 (Testing):** 2-3 hours

**Total Estimated Time:** 8-13 hours

## Success Criteria

1. Detection rate >60% (up from ~10%)
2. Hand detection rate >70% for frames with visible hands
3. Coordinate accuracy: >90% of keypoints within 5% error
4. No false body part detections when only hands visible
5. Hands in center of frame prioritized when multiple hands present

## Next Steps

1. Review and approve this plan
2. Implement Phase 1 fixes immediately
3. Test with debug script
4. Iterate based on results
5. Document final solution

