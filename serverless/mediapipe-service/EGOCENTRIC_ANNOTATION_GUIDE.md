# Egocentric Video Annotation Guide - MediaPipe Hands + Shoulders

## 🎯 Recommended Setup for Egocentric Videos

For annotating egocentric videos with **precise hands annotations up to shoulders**, you have two options:

### Option 1: "hands-shoulders-skeleton" (Recommended for Egocentric Videos) ⭐

This label is specifically designed for egocentric videos and includes:
- ✅ **Upper body keypoints**: shoulders, elbows, wrists (6 keypoints)
- ✅ **Hand keypoints**: all 5 fingers per hand (40 keypoints)
- ✅ **Total**: 46 keypoints with edges connecting them

**Advantages for egocentric videos:**
- 🎯 Focused on the visible parts (arms and hands)
- 🎯 No unnecessary keypoints (hips, knees, ankles are rarely visible)
- 🎯 Faster annotation and processing
- 🎯 Perfect for hand-object interaction studies

### Option 2: "person-skeleton" (Full Body)

This label includes the full body:
- ✅ **Body keypoints**: shoulders, elbows, wrists, hips, knees, ankles (17 keypoints)
- ✅ **Hand keypoints**: all 5 fingers per hand (40 keypoints)
- ✅ **Total**: 57 keypoints with edges connecting them

**Use this when:**
- You need full body tracking
- Lower body is visible in your videos
- You want comprehensive pose analysis

### ⚠️ Label Naming: "person-skeleton" vs "person"

The skeleton label is named **"person-skeleton"** (not "person") to avoid conflicts with semantic segmentation labels that also use "person". This allows you to have:
- **Semantic segmentation**: `"person"` label (for masks/regions)
- **Skeleton annotation**: `"person-skeleton"` label (for keypoints)

Both can coexist in the same CVAT project without conflicts!

## 📋 Two Methods to Import Skeleton Labels

### Method 1: Import from Model (Recommended) ⭐

**Best for**: When MediaPipe function is already deployed (which it is!)

**Steps**:
1. Open your CVAT project/task
2. Go to **Labels** → **Constructor** (or click "Setup" when creating a task)
3. Click **"From model"** button
4. Select **"MediaPipe Pose + Hands"** from the dropdown
5. Click **"Import"**
6. ✅ **All three labels will be imported**:
   - **"hands-shoulders-skeleton"**: 46 keypoints (shoulders + hands) - **recommended for egocentric videos**
   - **"person-skeleton"**: 57 keypoints (full body + hands) - for full body annotation
   - **"hands-skeleton"**: 42 keypoints (hands only) - for hands-only annotation

   You can choose which labels to keep in your project, or keep all three!

**Advantages**:
- ✅ Automatically matches the MediaPipe function specification
- ✅ No manual copy-paste needed
- ✅ Guaranteed compatibility with auto-annotation
- ✅ Includes all edges and connections

### Method 2: Copy-Paste from Raw Config

**Best for**: Manual setup or when you want to customize

**Steps**:
1. Open your CVAT project/task
2. Go to **Labels** → **Constructor** (or click "Setup" when creating a task)
3. Click **"Raw"** tab (Raw label editor)
4. Open `serverless/mediapipe-service/mediapipe-skeletons-raw-editor.json`
5. Copy **the labels you want** from the array:
   - **"person-skeleton"** (second object) - for body + hands
   - **"hands-skeleton"** (first object) - for hands only
   - Or copy both if you want both options
   ```json
   {
     "name": "person-skeleton",
     "type": "skeleton",
     "attributes": [],
     "svg": "...",
     "sublabels": [...]
   }
   ```
6. Wrap it in an array: `[{...}]`
7. Paste into the Raw editor
8. Click **"Done"**

**File location**: `serverless/mediapipe-service/mediapipe-skeletons-raw-editor.json`

## 🎨 What You'll See

After importing, you'll have:
- **Points**: 57 keypoints visible as circles
- **Edges**: Lines connecting:
  - Body skeleton (shoulders → elbows → wrists)
  - Hand skeleton (wrist → fingers with all joints)
  - All connections between related keypoints

## 🚀 Using Auto-Annotation

Once labels are imported:

1. **Create/Open a task** with your egocentric video
2. Go to **Actions** → **Run auto-annotation**
3. Select **"MediaPipe Pose + Hands"** model
4. Click **"Annotate"**
5. ✅ MediaPipe will automatically annotate **all matching labels** in your project:
   - If you have **"hands-shoulders-skeleton"** label: annotates shoulders + hands (46 keypoints)
   - If you have **"person-skeleton"** label: annotates body + hands (57 keypoints)
   - If you have **"hands-skeleton"** label: annotates hands only (42 keypoints)
   - If you have **multiple labels**: annotates all matching labels (you'll get multiple annotations)
   - All edges will be automatically drawn

### 🎯 Choosing Which Labels to Use

**Option 1: Use only "hands-shoulders-skeleton"** (recommended for egocentric videos) ⭐
- Import only "hands-shoulders-skeleton" from model
- Get shoulders, elbows, wrists, and hands annotation (46 keypoints)
- Best for egocentric videos where only upper body is visible
- Faster processing and cleaner annotations

**Option 2: Use only "person-skeleton"**
- Import only "person-skeleton" from model
- Get full body + hands annotation (57 keypoints)
- Best for comprehensive pose tracking when full body is visible

**Option 3: Use only "hands-skeleton"**
- Import only "hands-skeleton" from model
- Get hands-only annotation (42 keypoints)
- Best for hand-focused analysis without arm tracking

**Option 4: Use multiple labels**
- Import multiple labels from model
- Get different annotation granularities
- Useful for comparing or having flexible annotation options

## 📊 Skeleton Structure

### Hands-Shoulders Skeleton (46 keypoints) - Recommended for Egocentric Videos

**Upper Body Keypoints (6)**
- Shoulders: left_shoulder, right_shoulder
- Elbows: left_elbow, right_elbow
- Wrists: left_wrist, right_wrist

**Hand Keypoints (40 total, 20 per hand)**
Each hand has:
- Thumb: cmc, mcp, ip, tip (4 keypoints)
- Index: mcp, pip, dip, tip (4 keypoints)
- Middle: mcp, pip, dip, tip (4 keypoints)
- Ring: mcp, pip, dip, tip (4 keypoints)
- Pinky: mcp, pip, dip, tip (4 keypoints)

### Person Skeleton (57 keypoints) - Full Body

**Body Keypoints (17)**
- Face: nose, left_eye, right_eye, left_ear, right_ear
- Upper body: left_shoulder, right_shoulder, left_elbow, right_elbow, left_wrist, right_wrist
- Lower body: left_hip, right_hip, left_knee, right_knee, left_ankle, right_ankle

**Hand Keypoints (40 total, 20 per hand)**
Each hand has:
- Wrist (already in body)
- Thumb: cmc, mcp, ip, tip (4 keypoints)
- Index: mcp, pip, dip, tip (4 keypoints)
- Middle: mcp, pip, dip, tip (4 keypoints)
- Ring: mcp, pip, dip, tip (4 keypoints)
- Pinky: mcp, pip, dip, tip (4 keypoints)

### Hands-Only Skeleton (42 keypoints)

**Hand Keypoints (42 total, 21 per hand)**
Each hand has:
- Wrist
- Thumb: cmc, mcp, ip, tip (4 keypoints)
- Index: mcp, pip, dip, tip (4 keypoints)
- Middle: mcp, pip, dip, tip (4 keypoints)
- Ring: mcp, pip, dip, tip (4 keypoints)
- Pinky: mcp, pip, dip, tip (4 keypoints)

## ⚠️ Important Notes

1. **Name Matching**: Keypoint names must match exactly for auto-annotation to work
2. **Edges**: All edges are automatically included in the SVG
3. **Visualization**: Points and edges will be visible when annotating
4. **Compatibility**: The "person-skeleton" label matches the MediaPipe function spec exactly

## 🔍 Verification

After importing, verify:
- ✅ "hands-shoulders-skeleton" label exists (recommended for egocentric videos)
- ✅ "person-skeleton" label exists (distinct from semantic segmentation "person" label)
- ✅ "hands-skeleton" label exists (optional, for hands-only annotation)
- ✅ Correct number of sublabels are present (46 for hands-shoulders, 57 for person, 42 for hands)
- ✅ SVG preview shows all points and edges
- ✅ MediaPipe function appears in auto-annotation dropdown

## 💡 Tips for Egocentric Videos

- **Frame rate**: MediaPipe works best with videos at 15-30 FPS
- **Resolution**: Higher resolution = better hand detection
- **Lighting**: Ensure good lighting for accurate hand tracking
- **Occlusion**: Hands may be occluded in egocentric view - you can manually adjust annotations

