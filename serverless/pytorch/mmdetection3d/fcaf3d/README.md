# FCAF3D with MMDetection3D for CVAT

This directory contains the FCAF3D implementation using MMDetection3D for CVAT automatic annotation of indoor 3D point cloud datasets.

## Overview

FCAF3D (Fully Convolutional Anchor-Free 3D Object Detection) is a state-of-the-art method for indoor 3D object detection. This implementation uses MMDetection3D framework for seamless integration with CVAT.

## Features

- **State-of-the-art Performance**: 64.3% mAP on ScanNet V2 indoor dataset
- **Anchor-free Detection**: No prior assumptions about object geometry
- **GPU Acceleration**: ROCm support for AMD GPUs
- **CVAT Integration**: Ready-to-use with CVAT automatic annotation
- **Pedestrian Focus**: Optimized for social navigation pedestrian detection

## Requirements

- ROCm-compatible GPU (AMD Radeon series)
- Docker
- Nuclio (for CVAT deployment)

## Installation

### ⚠️ **Important: Model Checkpoint Required**

**The FCAF3D checkpoint cannot be downloaded automatically due to URL issues.** You must manually download the pre-trained weights for the model to function:

1. Visit: https://github.com/open-mmlab/mmdetection3d/blob/main/docs/en/model_zoo.md
2. Search for "fcaf3d" and download the ScanNet checkpoint (should be ~200MB .pth file)
3. Place it at: `checkpoints/fcaf3d_scannet.pth`
4. Then proceed with Docker build

The setup script will create a placeholder file and provide detailed instructions.

### ROCm GPU Setup
For modern AMD GPUs with ROCm 7.1+ support. This provides optimal performance for FCAF3D inference.

**GPU Requirements:**
- AMD Radeon GPU with ROCm 7.1+ support
- At least 8GB VRAM recommended
- GPU drivers and ROCm userspace installed on host

```bash
cd serverless/pytorch/mmdetection3d/fcaf3d

# Build ROCm container
docker build -f Dockerfile.rocm -t ankk98/cvat-fcaf3d-rocm:latest .

# Setup model
chmod +x setup_model.sh
./setup_model.sh

# Test setup
python3 test_setup.py

# Deploy to CVAT (ROCm mode)
cd ../../../../
FCAF3D_USE_ROCM=1 ./deploy_rocm_pointcloud_models.sh fcaf3d deploy
```

### Automated Setup Script

The setup script automatically handles model download and configuration:

```bash
cd serverless/pytorch/mmdetection3d/fcaf3d
./setup_model.sh  # Downloads model and creates config
python3 test_setup.py  # Validates setup
```

### Manual Setup

#### 1. Download Pre-trained Model

**Option A: Automatic Download (Recommended)**
```bash
# Run the setup script which tries multiple URLs
./setup_model.sh
```

**Option B: Manual Download**
```bash
# Create checkpoints directory
mkdir -p checkpoints
cd checkpoints

# Method 1: Direct download (if URL is known)
wget https://download.openmmlab.com/mmdetection3d/v1.0.0_models/fcaf3d/fcaf3d_8x_scannet-3d-20class_20220320_124030-600d644c.pth

# Method 2: From MMDetection3D GitHub releases
# Visit: https://github.com/open-mmlab/mmdetection3d/releases
# Look for fcaf3d model in the assets

# Method 3: Use MMDetection3D model zoo
# Visit: https://github.com/open-mmlab/mmdetection3d/tree/1.0/configs/fcaf3d
# Look for download links in README files

# Rename to expected filename:
mv fcaf3d_8x_scannet-3d-20class_20220320_124030-600d644c.pth fcaf3d_scannet.pth
```

#### 2. Build Appropriate Container

**Build ROCm Container:**
```bash
docker build -f Dockerfile.rocm -t ankk98/cvat-fcaf3d-rocm:latest .
```

#### 3. Test Setup

```bash
# Run validation tests
python3 test_setup.py
```

#### 4. Deploy to CVAT

**ROCm Deployment:**
```bash
cd ../../../../  # Back to serverless root
FCAF3D_USE_ROCM=1 ./deploy_rocm_pointcloud_models.sh fcaf3d deploy
```

**Note:** Make sure to push the built image to Docker Hub before deployment:
```bash
docker push ankk98/cvat-fcaf3d-rocm:latest
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DEFAULT_CONFIDENCE_THRESHOLD` | `0.3` | Detection confidence threshold |
| `FCAF3D_CONFIG` | `configs/fcaf3d/fcaf3d_1x8_scannet-3d-18class.py` | Model configuration file |
| `FCAF3D_CHECKPOINT` | `checkpoints/fcaf3d_scannet.pth` | Model checkpoint file |

### Model Configuration

The default configuration uses FCAF3D trained on ScanNet dataset. For custom training:

```python
# In your config file
model = dict(
    type='FCAF3D',
    backbone=dict(
        type='MinkResNet',
        in_channels=3,
        depth=34,
        norm='batch',
    ),
    neck=dict(
        type='FCAF3DNeck',
        in_channels=(64, 128, 256, 512),
        out_channels=128,
    ),
    bbox_head=dict(
        type='FCAF3DHead',
        in_channels=128,
        num_classes=18,  # ScanNet classes
        bbox_coder=dict(type='FCAF3DBBoxCoder'),
    ),
)
```

## Usage

### In CVAT

1. Open your 3D point cloud task
2. Go to Automatic Annotation
3. Select "FCAF3D 3D Cuboids (ROCm)" from the model dropdown
4. Adjust confidence threshold if needed
5. Click "Annotate"

### Expected Output

The model outputs CVAT-compatible cuboid annotations with:
- **Position**: Center coordinates (x, y, z)
- **Rotation**: Orientation angles (primarily Z-axis for heading)
- **Scale**: Dimensions (length, width, height)
- **Confidence**: Detection confidence score
- **Label**: 18 classes from ScanNet dataset

## Performance

| Dataset | mAP@0.5 | Notes |
|---------|---------|-------|
| ScanNet V2 | 64.3% | State-of-the-art indoor detection |
| Social Navigation | 60-65% | Estimated based on domain adaptation |

## Troubleshooting

### Common Issues

1. **ROCm GPU Errors**: Ensure AMD GPU drivers and ROCm 7.1+ are installed
2. **Device Access**: Verify `/dev/kfd` and `/dev/dri` are accessible
3. **Memory Issues**: Reduce batch size or use smaller point clouds (8GB+ VRAM recommended)
4. **Model Loading**: Verify checkpoint file exists and is compatible
5. **CVAT Integration**: Check Nuclio function logs for errors

### Debug Commands

```bash
# Check Nuclio function status
nuctl get function

# View function logs
nuctl logs -f pth-mmdet3d-fcaf3d-rocm

# Test inference locally (ROCm GPU access)
docker run --rm \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  -v $(pwd)/checkpoints:/opt/fcaf3d/checkpoints:ro \
  ankk98/cvat-fcaf3d-rocm:latest python3 -c "
import torch
from mmdet3d.apis import init_model
print('GPU available:', torch.cuda.is_available())
model = init_model('configs/fcaf3d/fcaf3d_1x8_scannet-3d-18class.py', 'checkpoints/fcaf3d_scannet.pth')
print('Model loaded successfully')
"
```

### Performance Tuning

#### Memory Optimization
- **Batch Size**: Reduce for limited GPU memory
- **Point Sampling**: Use `PointSample(num_points=20000)` for faster inference
- **Voxel Size**: Increase voxel size for coarser resolution

#### Accuracy Improvements
- **Multi-Scale Testing**: Enable test-time augmentation
- **Ensemble Methods**: Combine with SIT detector for better coverage
- **Fine-tuning**: Adapt on your specific indoor dataset

### Integration with Other Models

#### Ensemble with SIT
```python
# Combine FCAF3D with SIT for better pedestrian detection
# FCAF3D for high-precision detection
# SIT for motion estimation and tracking
```

#### Domain Adaptation
```python
# Use MS3D framework for transfer learning
# Adapt outdoor-trained FCAF3D to indoor environments
```

### Custom Training

#### Fine-tune on Your Dataset
```bash
# Prepare your indoor dataset in KITTI format
# Modify config for your classes (focus on pedestrian)
# Train with MMDetection3D
python tools/train.py configs/fcaf3d/fcaf3d_custom.py
```

## Architecture Details

### FCAF3D Pipeline

1. **Input Processing**: Point cloud voxelization
2. **Feature Extraction**: Sparse 3D convolutions (MinkowskiEngine)
3. **Detection Head**: Anchor-free prediction of 3D bounding boxes
4. **Post-processing**: Confidence thresholding and NMS

### CVAT Integration

- **Input Format**: Base64-encoded point cloud (.bin/.pcd)
- **Output Format**: CVAT cuboid JSON with 16-point format
- **Coordinate System**: Matches CVAT's 3D workspace conventions
- **Label Mapping**: ScanNet classes mapped to CVAT labels

## References

- [FCAF3D Paper](https://arxiv.org/abs/2112.00322)
- [MMDetection3D Documentation](https://mmdetection3d.readthedocs.io/)
- [CVAT Automatic Annotation](https://docs.cvat.ai/docs/manual/advanced/automatic-annotation/)

## Challenges and Learnings

During the implementation of FCAF3D for CVAT serverless functions, we encountered and resolved several critical technical challenges:

### 1. MinkowskiEngine CPU vs GPU Conflict
**Problem**: Docker build failed when compiling `MinkowskiEngine` with CUDA support due to resource limitations or compiler mismatch in the build environment.
**Part 1 Solution**: We compiled `MinkowskiEngine` in **CPU-only mode** (using a pre-built wheel) to ensure a successful build. This forces the heavy 3D sparse convolutions (Backbone/Neck) to run on the CPU.

### 2. MMCV NMS Device Mismatch
**Problem**: While the backbone runs on CPU (due to the above constraint), the Neural Network's post-processing (NMS - Non-Maximum Suppression) implemented in `mmcv-full` requires **GPU** tensors. Passing CPU tensors from the backbone to the GPU-only NMS function caused `RuntimeError: implementation for device cpu not found`.
**Part 2 Solution**: We implemented a hybrid execution model by monkey-patching `mmcv.ops.iou3d.nms3d_normal`.
- The model forces `device='cpu'` to satisfy MinkowskiEngine.
- The monkey-patched NMS wrapper intercepts the call, moves tensors to the GPU (if available) for the NMS operation, and then moves the results back to the CPU.
- This allows us to utilize the GPU for the computationally intensive sorting/IOU operations while keeping the backbone compatible with the CPU-only MinkowskiEngine build.

### 3. Deployment Structure
- **`nuclio/main.py`**: The entry point for the serverless function.
- **`main_modified.py`**: A local development/sync copy.
- **`function-rocm.yaml`**: Defines the serverless function configuration.
- **Service vs Function**: The deployment uses a "proxy" pattern where the Nuclio function forwards requests to a persistent web service (`fcaf3d-service`) to avoid initializing the heavy 3D model for every single serverless request.

## License

This implementation follows the licenses of FCAF3D and MMDetection3D.
