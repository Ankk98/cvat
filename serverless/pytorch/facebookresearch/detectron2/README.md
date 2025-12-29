# Detectron2 Models for CVAT

This directory contains Detectron2 model implementations for CVAT automatic annotation.

## Available Models

### 1. RetinaNet R101 - Object Detection
**Location:** `retinanet_r101/`

- **Model:** COCO-Detection/retinanet_R_101_FPN_3x
- **Output:** Bounding boxes (rectangles)
- **Use Case:** Fast object detection without segmentation masks
- **Deployment:** `function-rocm.yaml` (ROCm GPU), `function-gpu.yaml` (CUDA GPU), `function.yaml` (CPU)

### 2. Mask R-CNN R50 ROCm - Instance Segmentation ✅ PRODUCTION READY
**Location:** `mask_rcnn_r50_rocm/`

- **Model:** COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x
- **Output:** Instance segmentation masks (flattened pixels format, converted to RLE by CVAT backend)
- **Use Case:** Precise object segmentation with pixel-level masks for egocentric videos
- **Deployment:** `function-rocm.yaml` (ROCm GPU)
- **Status:** ✅ **Production Ready** - Masks render correctly in CVAT, tested with egocentric videos
- **Format:** Provides flattened mask pixels `[pixel1, pixel2, ..., x_min, y_min, x_max, y_max]` which CVAT backend converts to RLE


## Configuration Files

**Location:** `configs/`

- `detectron2-coco-masks.json` - Full COCO dataset labels (80 classes) for mask annotation
- `detectron2-egocentric-masks.json` - Egocentric/kitchen-focused labels (36 classes) for mask annotation
- `DETECTRON2_MASKS_CONFIG.md` - Documentation for mask configuration files
- `CVAT_SEMANTIC_MASKS_SURVEY.md` - CVAT semantic mask handling survey

## Deployment

### RetinaNet R101 (Object Detection)

```bash
cd serverless/pytorch/facebookresearch/detectron2/retinanet_r101/nuclio
nuctl deploy --project-name cvat \
  --path . \
  --file function-rocm.yaml \
  --platform local \
  --env CVAT_FUNCTIONS_REDIS_HOST=cvat_redis_ondisk \
  --env CVAT_FUNCTIONS_REDIS_PORT=6666 \
  --platform-config '{"attributes": {"network": "cvat_cvat"}}'
```

### Mask R-CNN R50 ROCm (Instance Segmentation)

```bash
cd serverless/pytorch/facebookresearch/detectron2/mask_rcnn_r50_rocm/nuclio
nuctl deploy --project-name cvat \
  --path . \
  --file function-rocm.yaml \
  --platform local \
  --env CVAT_FUNCTIONS_REDIS_HOST=cvat_redis_ondisk \
  --env CVAT_FUNCTIONS_REDIS_PORT=6666 \
  --platform-config '{"attributes": {"network": "cvat_cvat"}}'
```

## Testing

See `serverless/test-scripts/` for test scripts:
- `test_detectron2_masks.py` - Test mask output format
- `test_detectron2_egocentric.py` - Test egocentric vision scenarios

## Differences

| Model | Output Type | Format | Use Case |
|-------|------------|--------|----------|
| RetinaNet R101 | Bounding boxes | `[x1, y1, x2, y2]` | Fast object detection |
| Mask R-CNN R50 | Segmentation masks | `[pixel1, pixel2, ..., x_min, y_min, x_max, y_max]` | Precise segmentation |

**Note:** Mask R-CNN provides flattened mask pixels, which CVAT backend converts to RLE format automatically.

## Notes

- **RetinaNet** outputs rectangles (bounding boxes) - faster, less precise
- **Mask R-CNN** outputs masks (segmentation) - slower, more precise
- Both models use COCO dataset categories
- Mask R-CNN provides flattened pixels format (matches OpenVINO `to_cvat_mask()`), CVAT backend converts to RLE
- ✅ **Mask R-CNN is production ready** - Masks render correctly in CVAT canvas

