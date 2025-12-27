# CVAT Serverless Models & Deployment

This directory contains the serverless model implementations, deployment scripts, and strategic plans for CVAT's auto-annotation capabilities. The focus is on enabling efficient computer vision model deployment for annotation workflows, particularly targeting egocentric vision tasks.

## 📁 Directory Structure

### Deployment Scripts & Tools
| Script | Purpose | Key Features |
|--------|---------|--------------|
| [`deploy_egocentric_models.sh`](./deploy_egocentric_models.sh) | **Primary deployment tool** for egocentric vision models | Automated ROCm/CPU deployment, selective model deployment, comprehensive logging |
| [`deploy_rocm_host.sh`](./deploy_rocm_host.sh) | ROCm deployment on host system | AMD GPU acceleration, automatic config detection |
| [`deploy_rocm_toolbox.sh`](./deploy_rocm_toolbox.sh) | ROCm deployment in toolbox containers | Isolated environments, AMD GPU support |
| [`deploy_cpu.sh`](./deploy_cpu.sh) | CPU-only model deployment | Legacy models, resource-constrained environments |
| [`deploy_gpu.sh`](./deploy_gpu.sh) | NVIDIA GPU deployment | CUDA acceleration, high-performance inference |

### Specialized Deployment Scripts
- [`deploy_rocm_robotics_models.sh`](./deploy_rocm_robotics_models.sh) - Curated robotics/VLA models
- [`deploy_rocm_pointcloud_models.sh`](./deploy_rocm_pointcloud_models.sh) - 3D point cloud processing models

## 🤖 Model Implementations

### Independent Services

#### MediaPipe Pose Service ([`mediapipe-service/`](./mediapipe-service/))
**Complete standalone pose detection service** - Since MediaPipe has compatibility issues with Nuclio, we've created a full-featured independent service.

**Features:**
- **33-point pose estimation** with MediaPipe
- **Hand-focused filtering** for egocentric vision
- **FastAPI-based REST API** with automatic CVAT integration
- **Multiple deployment options**: Virtual environment, Docker, Docker Compose
- **Automatic setup scripts** with CVAT function registration

**Quick Start:**
```bash
cd mediapipe-service
./run-setup.sh --cvat-url http://localhost:8080
```

**API Endpoints:**
- `GET /health` - Service health check
- `POST /detect` - Pose detection with CVAT-compatible output
- `GET /` - Service information and documentation

**Status:** ✅ **Fully Working** - Ready for production use

---

### PyTorch Models (`pytorch/`)

#### Core Egocentric Vision Models
| Model | Path | Purpose | ROCm Support | Status |
|-------|------|---------|--------------|--------|
| **SAM** | [`facebookresearch/sam/`](./pytorch/facebookresearch/sam/) | Interactive segmentation for hands/objects | ✅ Full | ✅ Working |
| **Detectron2** | [`facebookresearch/detectron2/retinanet_r101/`](./pytorch/facebookresearch/detectron2/retinanet_r101/) | Instance segmentation | ✅ Full | ✅ Working |
| **MMPose** | [`mmpose/hrnet32/`](./pytorch/mmpose/hrnet32/) | Whole-body pose estimation | ❌ CPU only | ⚠️ Needs fixes |
| **YOLO11 Pose** | [`ultralytics/yolov11-pose/`](./pytorch/ultralytics/yolov11-pose/) | Real-time pose estimation | ❌ CPU only | ⚠️ Needs fixes |
| **MediaPipe Pose + Hands** | [`mediapipe-service/`](./mediapipe-service/) | 75-point pose + finger joint tracking | ✅ **Working** | 🚀 Independent service |

#### Advanced Models
| Model | Path | Purpose | ROCm Support | Status |
|-------|------|---------|--------------|--------|
| **Transt** | [`dschoerk/transt/`](./pytorch/dschoerk/transt/) | Visual object tracking | ✅ Full | ✅ Working |
| **SiamMask** | [`foolwood/siammask/`](./pytorch/foolwood/siammask/) | Object tracking with segmentation | ✅ GPU | ✅ Working |
| **IOG** | [`shiyinzhang/iog/`](./pytorch/shiyinzhang/iog/) | Interactive object segmentation | ❌ CPU | ✅ Working |
| **Open3D SIT** | [`open3d/sit_pointcloud/`](./pytorch/open3d/sit_pointcloud/) | 3D point cloud segmentation | ✅ Full | ✅ Working |
| **MediaPipe Pose** | [`google/mediapipe-pose/`](./pytorch/google/mediapipe-pose/) | Lightweight 11-point pose estimation | ❌ Complex | 📋 Reference |

### OpenVINO Models (`openvino/`)

#### Intel Open Model Zoo Models
| Model | Path | Purpose | Acceleration |
|-------|------|---------|--------------|
| **Semantic Segmentation ADAS** | [`omz/intel/semantic-segmentation-adas-0001/`](./openvino/omz/intel/semantic-segmentation-adas-0001/) | Road scene segmentation | CPU/OpenVINO |
| **Face Detection** | [`omz/intel/face-detection-0205/`](./openvino/omz/intel/face-detection-0205/) | Face detection | CPU/OpenVINO |
| **Person ReID** | [`omz/intel/person-reidentification-retail-0277/`](./openvino/omz/intel/person-reidentification-retail-0277/) | Person identification | CPU/OpenVINO |
| **Text Detection** | [`omz/intel/text-detection-0004/`](./openvino/omz/intel/text-detection-0004/) | Text region detection | CPU/OpenVINO |

#### Public Models
| Model | Path | Purpose | Framework |
|-------|------|---------|-----------|
| **Mask R-CNN** | [`omz/public/mask_rcnn_inception_resnet_v2_atrous_coco/`](./openvino/omz/public/mask_rcnn_inception_resnet_v2_atrous_coco/) | Instance segmentation | TensorFlow |
| **Faster R-CNN** | [`omz/public/faster_rcnn_inception_resnet_v2_atrous_coco/`](./openvino/omz/public/faster_rcnn_inception_resnet_v2_atrous_coco/) | Object detection | TensorFlow |
| **YOLOv3** | [`omz/public/yolo-v3-tf/`](./openvino/omz/public/yolo-v3-tf/) | Object detection | TensorFlow |

### Other Frameworks
| Framework | Path | Models | Status |
|-----------|------|--------|--------|
| **ONNX** | [`onnx/`](./onnx/) | YOLOv7 | Legacy |
| **TensorFlow** | [`tensorflow/`](./tensorflow/) | Faster R-CNN | Legacy |

## 📋 Strategic Plans & Roadmaps

### Core Implementation Plans

#### 🚀 [PLAN_SOTA_MODELS.md](./PLAN_SOTA_MODELS.md)
**State-of-the-Art Model Integration Plan**
- Timeline: 9 months implementation
- Focus: Latest computer vision models for egocentric vision
- Key models: YOLO11, BEiT3, OneFormer, DETRPose, MaskDINO
- Target: 35-45% accuracy improvements

**Phases:**
1. **Phase 1 (Months 1-4)**: YOLO11 Pose, BEiT3, OneFormer
2. **Phase 2 (Months 4-7)**: DETRPose, MaskDINO, MediaPipe
3. **Phase 3 (Months 7-9)**: Advanced optimizations

#### ⚡ [PLAN_AMD_GPU_ACCELERATION.md](./PLAN_AMD_GPU_ACCELERATION.md)
**AMD GPU Acceleration Strategy**
- Timeline: 9 months implementation
- Focus: Maximize AMD GPU performance for CVAT workloads
- Technologies: ROCm, mixed precision, memory optimization
- Target: 5-15x performance improvements

**Phases:**
1. **Phase 1 (Months 1-3)**: ROCm infrastructure, MMPose GPU integration
2. **Phase 2 (Months 3-5)**: Quantization, multi-model pipelines, YOLO integration
3. **Phase 3 (Months 5-8)**: Mixed precision, memory optimization, real-time features
4. **Phase 4 (Months 8-9)**: Production deployment, monitoring

#### 🔧 [PLAN_MMPose_YOLO11_CPU_DEPLOYMENT.md](./PLAN_MMPose_YOLO11_CPU_DEPLOYMENT.md)
**CPU Deployment Fixes for Complex Models**
- Timeline: 6 weeks implementation
- Focus: Enable MMPose and YOLO11 on CPU systems
- Challenges: Python runtime conflicts, dependency management
- Target: Working CPU deployment for pose estimation

**Phases:**
1. **Phase 1 (Weeks 1-2)**: MMPose fixes, YOLO11 setup
2. **Phase 2 (Weeks 3-4)**: Optimization and container builds
3. **Phase 3 (Weeks 5-6)**: Testing and production deployment

### Quick Wins Status

| Model | Current Status | Quick Fix Available | ETA |
|-------|----------------|-------------------|-----|
| **SAM** | ✅ Working | N/A | Now |
| **Detectron2** | ✅ Working | N/A | Now |
| **MMPose** | ❌ Broken | Python 3.9 upgrade | 1 week |
| **YOLO11 Pose** | ❌ Broken | CPU deployment setup | 2 weeks |
| **MediaPipe Pose + Hands** | ✅ **Working** | Comprehensive finger joint tracking | Now |

## 🛠️ Development Guides

### [ROCM_TOOLBOX_GUIDE.md](./ROCM_TOOLBOX_GUIDE.md)
**AMD GPU Deployment Guide**
- ROCm toolbox setup and usage
- AMD GPU optimization strategies
- Performance benchmarking
- Troubleshooting common issues

### Deployment Best Practices

#### For Egocentric Vision Models
```bash
# Deploy all working egocentric models
./deploy_egocentric_models.sh

# Deploy specific models
./deploy_egocentric_models.sh --sam --detectron2

# Force CPU deployment
./deploy_egocentric_models.sh --cpu --mmpose
```

#### For AMD GPU Acceleration
```bash
# ROCm deployment (recommended for AMD GPUs)
./deploy_rocm_host.sh pytorch/facebookresearch/sam/
./deploy_rocm_host.sh pytorch/facebookresearch/detectron2/retinanet_r101/

# Robotics models bundle
./deploy_rocm_robotics_models.sh
```

## 📊 Model Performance Matrix

### Egocentric Vision Tasks

| Task | Current Best | Target SOTA | Accuracy Gain | Status |
|------|-------------|-------------|---------------|--------|
| **Interactive Segmentation** | SAM | SAM 2.0 | +10-20% | ✅ Working |
| **Instance Segmentation** | Mask R-CNN | BEiT3/MaskDINO | +35-45% | ✅ Working |
| **Semantic Segmentation** | ADAS Model | OneFormer | Massive | ❌ Needs implementation |
| **Pose Estimation** | MediaPipe Pose + Hands | YOLO11/DETRPose | +0.4-5% | ✅ **Complete finger tracking** |
| **Object Detection** | YOLOv7 | YOLO11 | +5-10% | ❌ Needs implementation |

### Hardware Acceleration

| Hardware | Current Support | Target Performance | Status |
|----------|----------------|-------------------|--------|
| **AMD GPU (ROCm)** | Partial (3 models) | Full acceleration | ✅ Working models + plans |
| **NVIDIA GPU** | Full (CUDA) | Optimized pipelines | ✅ Existing |
| **CPU** | Full | Optimized inference | ⚠️ Needs fixes for complex models |

## 🚀 Quick Start

### Deploy Working Egocentric Models
```bash
# Navigate to serverless directory
cd serverless

# Deploy all currently working egocentric models
./deploy_egocentric_models.sh

# Check deployment status
nuctl get functions
```

### Expected Output
```
 NAMESPACE | NAME                                | PROJECT | STATE | REPLICAS | NODE PORT
 nuclio    | pth-facebookresearch-sam-vit-h-rocm | cvat    | ready | 1/1      | 32768
 nuclio    | pth-facebookresearch-detectron2-retinanet-r101-rocm | cvat    | ready | 1/1      | 32769
```

### Test the Models
```bash
# Test SAM segmentation
curl -X POST http://localhost:32768 \
  -H "Content-Type: application/json" \
  -d '{"image": "base64_encoded_image_data"}'

# Test Detectron2 instance segmentation
curl -X POST http://localhost:32769 \
  -H "Content-Type: application/json" \
  -d '{"image": "base64_encoded_image_data"}'
```

## 🔧 Troubleshooting

### Common Issues

#### Model Deployment Failures
```bash
# Check Nuclio function status
nuctl get functions

# View function logs
nuctl logs -f <function-name> --platform local

# Delete failed function
nuctl delete function <function-name> --platform local
```

#### ROCm Issues on AMD GPUs
```bash
# Check ROCm installation
rocm-smi

# Verify GPU detection
python -c "import torch; print(torch.cuda.is_available())"

# Use toolbox deployment for isolation
./deploy_rocm_toolbox.sh pytorch/facebookresearch/sam/ my-toolbox
```

#### CPU Deployment Issues
```bash
# Force CPU deployment
./deploy_egocentric_models.sh --cpu --all

# Check CPU model compatibility
python -c "import torch; print('CPU available:', torch.cuda.is_available() == False)"
```

## 📈 Monitoring & Performance

### Key Metrics to Track
- **Deployment success rate**: >95% target
- **Inference latency**: <500ms per image
- **GPU utilization**: >80% on AMD GPUs
- **Memory usage**: <8GB for large models
- **Accuracy preservation**: >95% of original

### Performance Benchmarking
```bash
# Run performance tests
python -c "
import time
# Test model inference speed
# Compare CPU vs GPU performance
# Measure memory usage
"
```

## 🤝 Contributing

### Adding New Models
1. **Choose appropriate framework directory** (`pytorch/`, `openvino/`, etc.)
2. **Create model subdirectory** with `nuclio/` folder
3. **Implement** `function.yaml`, `function-rocm.yaml`, `main.py`
4. **Add ROCm support** for AMD GPU acceleration
5. **Update deployment scripts** if needed
6. **Document** in this index and relevant plans

### Testing Requirements
- [ ] CPU deployment works
- [ ] ROCm deployment works (AMD GPUs)
- [ ] Model accuracy validated
- [ ] Performance benchmarks completed
- [ ] Documentation updated

## 📚 Additional Resources

- **CVAT Documentation**: Main project documentation
- **Nuclio Documentation**: Serverless function framework
- **ROCm Documentation**: AMD GPU acceleration
- **OpenVINO Documentation**: Intel model optimization

---

**Last Updated**: December 27, 2025
**Maintained by**: CVAT Development Team
**Contact**: For issues with serverless models or deployment
