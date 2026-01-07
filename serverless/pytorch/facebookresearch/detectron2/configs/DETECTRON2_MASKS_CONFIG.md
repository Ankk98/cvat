# Detectron2 Semantic Masks Configuration

## Overview

CVAT label configuration files for Detectron2 semantic segmentation masks. These configs can be imported into CVAT projects to use with Detectron2 Mask R-CNN for automatic annotation.

## Files

### 1. `detectron2-coco-masks.json` (Full COCO Dataset)

**Purpose:** Complete COCO dataset configuration with all 80 classes.

**Use case:** When you need comprehensive object detection across all COCO categories.

**Classes:** All 80 COCO categories including:
- People, vehicles, animals
- Food items, kitchenware, furniture
- Electronics, sports equipment
- And more...

**Total labels:** 80 classes

### 2. `detectron2-egocentric-masks.json` (Egocentric-Focused)

**Purpose:** Focused configuration for egocentric vision scenarios (kitchen, hands, daily activities).

**Use case:** Optimized for egocentric video annotation where you primarily encounter:
- Hands and people
- Kitchen items and food
- Common household objects
- Electronics and tools

**Classes included:**
- **Person** (id: 1)
- **Kitchen items:** bottle, wine glass, cup, fork, knife, spoon, bowl, microwave, oven, toaster, sink, refrigerator
- **Food:** banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake
- **Furniture:** chair, couch, dining table
- **Electronics:** tv, laptop, mouse, remote, keyboard, cell phone
- **Other:** handbag, tie, potted plant, book, clock, scissors

**Total labels:** 36 classes

## Usage

### Import into CVAT Project

1. **Create a new CVAT project** or open an existing one
2. **Go to Labels** section
3. **Click "Import"** or "Add Labels"
4. **Select the JSON file** (`detectron2-coco-masks.json` or `detectron2-egocentric-masks.json`)
5. **Verify labels** are imported correctly

### Use with Detectron2 Auto-Annotation

1. **Deploy Detectron2 service** (if not already deployed):
   ```bash
   cd serverless/pytorch/facebookresearch/detectron2/retinanet_r101/nuclio
   nuclio deploy -p detectron2-mask-rcnn
   ```

2. **In CVAT:**
   - Go to your task/project
   - Click "Actions" → "Run auto annotation"
   - Select Detectron2 function
   - Configure threshold (default: 0.5)
   - Run annotation

3. **Verify results:**
   - Check that masks are generated correctly
   - Verify label names match Detectron2 output
   - Adjust confidence threshold if needed

## Label ID Mapping

The label IDs in these configs match COCO category IDs used by Detectron2:

- **ID 1:** person
- **ID 44:** bottle
- **ID 47:** cup
- **ID 48:** fork
- **ID 49:** knife
- **ID 50:** spoon
- **ID 51:** bowl
- **ID 77:** cell phone
- **ID 78:** microwave
- **ID 79:** oven
- **ID 81:** sink
- **ID 82:** refrigerator
- ... (see full list in JSON files)

**Important:** Detectron2 returns COCO category IDs, so the label IDs in CVAT must match for proper mapping.

## Integration with Hand Pose Estimation

You can use both configs together:

1. **Hand pose skeleton:** `mediapipe-hands-skeleton.json`
   - For detailed hand keypoint tracking
   - 42 keypoints per hand

2. **Semantic masks:** `detectron2-egocentric-masks.json` (recommended)
   - For object detection and segmentation
   - Helps identify objects hands interact with

**Workflow:**
1. Import both configs into your CVAT project
2. Run MediaPipe for hand pose estimation
3. Run Detectron2 for semantic segmentation
4. Analyze intersection between hand keypoints and object masks

## Customization

### Add Custom Classes

To add custom classes not in COCO:

1. Edit the JSON file
2. Add new label entries:
   ```json
   {"id": 91, "name": "custom_object", "type": "mask", "attributes": []}
   ```
3. **Note:** You'll need a custom-trained Detectron2 model to detect custom classes

### Filter Classes

To use only specific classes:

1. Copy the full config
2. Remove unwanted classes
3. Keep only the classes you need
4. Re-import into CVAT

## Troubleshooting

### Labels Not Matching

**Problem:** Detectron2 detects objects but labels don't match.

**Solution:**
- Verify label IDs match COCO category IDs
- Check label names match exactly (case-sensitive)
- Ensure Detectron2 is using COCO model (not custom)

### Missing Masks

**Problem:** Detectron2 returns bounding boxes but no masks.

**Solution:**
- Ensure you're using Mask R-CNN model (not RetinaNet)
- Check Detectron2 service is configured for mask output
- Verify `type: "mask"` in label config

### Low Detection Rate

**Problem:** Detectron2 detects few objects.

**Solution:**
- Lower confidence threshold (try 0.3-0.4)
- Check image quality and resolution
- Verify objects are in COCO dataset
- Consider using custom-trained model for domain-specific objects

## File Format

Both configs use CVAT's standard label format:

```json
[
  {
    "id": 1,
    "name": "person",
    "type": "mask",
    "attributes": []
  },
  ...
]
```

- **id:** Numeric ID matching COCO category ID
- **name:** Class name (must match Detectron2 output)
- **type:** "mask" for semantic segmentation masks
- **attributes:** Optional array for label attributes

## See Also

- `mediapipe-hands-skeleton.json` - Hand pose skeleton configuration
- Detectron2 service: `serverless/pytorch/facebookresearch/detectron2/`
- CVAT documentation: https://docs.cvat.ai/

