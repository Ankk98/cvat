# MediaPipe Person Skeleton Configuration ✅ WORKING

This file contains the complete skeleton configuration for MediaPipe pose and hand detection, perfectly mapped to the CVAT MediaPipe auto-annotation function.

**Status**: ✅ **Fully integrated and working in CVAT auto-annotation!**

## 📁 Files

- **`mediapipe-person-skeleton.json`**: Complete skeleton configuration for copy-paste import

## 🎯 Skeleton Overview

**Total Keypoints**: 57 (14 pose + 43 hand keypoints)

### Pose Keypoints (14)
- **Face**: nose, left_eye, right_eye, left_ear, right_ear
- **Upper Body**: left_shoulder, right_shoulder, left_elbow, right_elbow
- **Lower Body**: left_hip, right_hip, left_knee, right_knee, left_ankle, right_ankle

### Hand Keypoints (43)
**Left Hand (21 keypoints)**:
- Wrist: `left_wrist`
- Thumb: `left_thumb_cmc`, `left_thumb_mcp`, `left_thumb_ip`, `left_thumb_tip`
- Index: `left_index_mcp`, `left_index_pip`, `left_index_dip`, `left_index_tip`
- Middle: `left_middle_mcp`, `left_middle_pip`, `left_middle_dip`, `left_middle_tip`
- Ring: `left_ring_mcp`, `left_ring_pip`, `left_ring_dip`, `left_ring_tip`
- Pinky: `left_pinky_mcp`, `left_pinky_pip`, `left_pinky_dip`, `left_pinky_tip`

**Right Hand (22 keypoints)**: Same structure with `right_` prefix

## 🚀 How to Use

### Method 1: Copy-Paste Import (Recommended)
1. **Open CVAT** and go to your project/task
2. **Go to Labels** → **Add Label**
3. **Set label name** to `person`
4. **Set type** to `Skeleton`
5. **Copy the entire JSON** from `mediapipe-person-skeleton.json`
6. **Paste into the skeleton editor**
7. **Save the label**

### Method 2: Manual Creation
If you prefer to create manually:
1. Create a `person` label with type `Skeleton`
2. Add all 57 keypoints listed above
3. Ensure names match exactly for MediaPipe compatibility

## ⚠️ Important Notes

- **Exact Name Matching**: Keypoint names must match exactly for auto-annotation to work
- **Order Matters**: The order in the JSON matches the MediaPipe function specification
- **Attributes Required**: Each keypoint includes an empty attributes array as required by CVAT
- **Ready for MediaPipe**: This skeleton is perfectly mapped to the MediaPipe auto-annotation function

## 🔗 Integration with MediaPipe Function

This skeleton is perfectly mapped to the `pth-google-mediapipe-pose-hands` function which provides:
- ✅ **Real-time pose detection** (33 MediaPipe pose keypoints)
- ✅ **Detailed hand tracking** (42 hand keypoints per detection)
- ✅ **Egocentric video optimization**
- ✅ **CVAT auto-annotation integration**

## 📊 Keypoint Mapping

| MediaPipe Index | Keypoint Name | Type |
|----------------|---------------|------|
| 0 | nose | pose |
| 1-4 | left_eye, right_eye, left_ear, right_ear | face |
| 5-8 | left_shoulder, right_shoulder, left_elbow, right_elbow | upper body |
| 9-14 | left_hip, right_hip, left_knee, right_knee, left_ankle, right_ankle | lower body |
| 15-35 | left_wrist + 20 left hand keypoints | left hand |
| 36-56 | right_wrist + 20 right hand keypoints | right hand |

## 🎉 Ready to Use!

Simply copy the JSON from `mediapipe-person-skeleton.json` and paste it into your CVAT skeleton label configuration. This will give you a complete pose + hand skeleton that works perfectly with the MediaPipe auto-annotation function!
