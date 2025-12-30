# Combined Config Compatibility: MediaPipe Hands + Detectron2 Masks

## Overview

This document confirms that MediaPipe hands skeleton (generated from `generate_skeleton_configs.py`) and `detectron2-egocentric-masks.json` can be used together in the same CVAT project without conflicts.

**Note:** To generate `mediapipe-hands-skeleton.json`, run:
```bash
python3 generate_skeleton_configs.py
```

## Compatibility Analysis

### ✅ Label Names: **NO CONFLICTS**

- **Hands skeleton:** Main label name is `"hands"` (type: skeleton)
- **Detectron2 masks:** Label names are COCO categories (`"person"`, `"bottle"`, `"cup"`, etc.)
- **Result:** No name conflicts ✅

### ✅ Label Types: **COMPATIBLE**

- **Hands skeleton:** `type: "skeleton"` with sublabels (keypoints)
- **Detectron2 masks:** `type: "mask"` (semantic segmentation masks)
- **Result:** Different types, fully compatible ✅

### ✅ Label IDs: **NO CONFLICTS**

**Important:** CVAT maps labels by **name**, not by ID. When importing label configs, CVAT removes the `id` fields and assigns new sequential IDs automatically.

**Skeleton Sublabel IDs:**
- Range: 0-41 (42 hand keypoints)
- Used for: SVG `data-label-id` references (must match sublabel IDs)
- Namespace: Sublabels (children of skeleton label)

**Detectron2 Mask Label IDs:**
- Values: 1, 31, 32, 44, 46, 47, ... (COCO category IDs)
- Used for: COCO category reference (for developers)
- Namespace: Top-level labels

**Overlap:** IDs 1, 31, 32 appear in both, but:
- ✅ CVAT removes IDs on import and reassigns
- ✅ Sublabels are in different namespace than top-level labels
- ✅ IDs are only used for reference/ordering in config files

**Result:** No conflicts ✅

## Usage

### Import Both Configs into CVAT Project

1. **Create or open CVAT project**
2. **Import hands skeleton:**
   - First, generate `mediapipe-hands-skeleton.json` by running `python3 generate_skeleton_configs.py`
   - Go to Labels → Import
   - Select `mediapipe-hands-skeleton.json`
   - Verify skeleton "hands" is created with 42 sublabels

3. **Import Detectron2 masks:**
   - Go to Labels → Import (or Add Labels)
   - Select `detectron2-egocentric-masks.json`
   - Verify 38 mask labels are added

4. **Verify no conflicts:**
   - Check that both "hands" skeleton and mask labels exist
   - Verify label names are unique

### Combined Workflow

1. **Run MediaPipe auto-annotation:**
   - Select "MediaPipe Pose + Hands" function
   - Annotates hand keypoints (skeleton type)

2. **Run Detectron2 auto-annotation:**
   - Select Detectron2 Mask R-CNN function
   - Annotates object masks (mask type)

3. **Analyze results:**
   - Hand keypoints from MediaPipe
   - Object masks from Detectron2
   - Intersection analysis (hands interacting with objects)

## Technical Details

### CVAT Label Import Process

Based on CVAT source code (`cvat/apps/engine/serializers.py`):

1. **ID Removal:** CVAT removes `id` fields from imported labels:
   ```python
   if label.get('id', None):
       del label['id']
   ```

2. **ID Reassignment:** CVAT assigns new sequential IDs automatically

3. **Name-Based Matching:** CVAT matches labels by name for auto-annotation:
   ```python
   ds_labels_by_name = {ds_label.name: ds_label for ds_label in ds_labels}
   ds_label = ds_labels_by_name.get(label_nm.name)
   ```

### Skeleton Sublabel IDs

- Sublabel IDs in config are used for SVG `data-label-id` validation
- Must match between SVG and sublabels array
- CVAT reassigns IDs on import, but SVG references are updated automatically

### Detectron2 Label IDs

- IDs match COCO category IDs (for reference)
- CVAT removes IDs on import
- Label names must match Detectron2 output exactly (case-sensitive)

## Verification Checklist

- [x] No label name conflicts
- [x] Different label types (skeleton vs mask)
- [x] CVAT handles IDs separately (removes and reassigns)
- [x] Sublabels in different namespace than top-level labels
- [x] Both configs can be imported into same project

## Conclusion

✅ **Both configs are fully compatible and can be used together in the same CVAT project.**

The ID overlap (1, 31, 32) is not a problem because:
1. CVAT removes IDs on import and reassigns them
2. Sublabels are in a different namespace than top-level labels
3. CVAT maps labels by name, not by ID

No changes needed to either config file.

