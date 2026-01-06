#!/bin/bash
# Setup script for FCAF3D MMDetection3D model
# Downloads pre-trained checkpoints and sets up configurations

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECKPOINT_DIR="${SCRIPT_DIR}/checkpoints"
CONFIG_DIR="${SCRIPT_DIR}/configs"

echo "Setting up FCAF3D model for MMDetection3D..."

# Create directories
mkdir -p "${CHECKPOINT_DIR}"
mkdir -p "${CONFIG_DIR}"

# Download FCAF3D ScanNet checkpoint
echo "Downloading FCAF3D ScanNet checkpoint..."

POSSIBLE_URLS=(
    "https://download.openmmlab.com/mmdetection3d/v1.0.0_models/fcaf3d/fcaf3d_8x2_scannet-3d-18class/fcaf3d_8x2_scannet-3d-18class_20220805_084956.pth"
)

CHECKPOINT_FILE="${CHECKPOINT_DIR}/fcaf3d_scannet.pth"

if [ ! -f "${CHECKPOINT_FILE}" ]; then
    echo "Trying to download FCAF3D checkpoint..."

    DOWNLOADED=false
    for CHECKPOINT_URL in "${POSSIBLE_URLS[@]}"; do
        echo "Trying URL: ${CHECKPOINT_URL}"
        if curl -L -s -o "${CHECKPOINT_FILE}" "${CHECKPOINT_URL}"; then
            FILE_SIZE=$(stat -c%s "${CHECKPOINT_FILE}" 2>/dev/null || echo "0")
            if [ "${FILE_SIZE}" -gt 100000000 ]; then  # 100MB
                echo "✓ Checkpoint downloaded successfully (${FILE_SIZE} bytes)"
                DOWNLOADED=true
                break
            else
                echo "⚠ Downloaded file too small, trying next URL..."
                rm -f "${CHECKPOINT_FILE}"
            fi
        else
            echo "✗ Failed to download from ${CHECKPOINT_URL}, trying next..."
        fi
    done

    if [ "$DOWNLOADED" = false ]; then
        echo "⚠ Warning: Could not download checkpoint automatically"
        echo ""
        echo "Please manually download FCAF3D ScanNet checkpoint:"
        echo "1. Visit: https://github.com/open-mmlab/mmdetection3d/tree/1.0/configs/fcaf3d"
        echo "2. Look for the model download link (usually in README or config files)"
        echo "3. Or visit: https://github.com/open-mmlab/mmdetection3d/blob/main/docs/en/model_zoo.md"
        echo "4. Search for 'fcaf3d_1x8_scannet-3d-18class' and download the .pth file"
        echo "5. Place the downloaded file at: ${CHECKPOINT_FILE}"
        echo ""
        echo "Expected filename: fcaf3d_8x2_scannet-3d-18class_20220805_084956.pth"
        echo "Expected size: ~200MB"
        echo ""
        echo "Direct download link:"
        echo "https://download.openmmlab.com/mmdetection3d/v1.0.0_models/fcaf3d/fcaf3d_8x2_scannet-3d-18class/fcaf3d_8x2_scannet-3d-18class_20220805_084956.pth"
        echo ""
        echo "For now, creating a placeholder file for testing..."
        touch "${CHECKPOINT_FILE}"
        echo "⚠ Using placeholder checkpoint - model will not work without real weights"
    fi
else
    echo "Checkpoint already exists at ${CHECKPOINT_FILE}"
fi

# Verify checkpoint
if [ -f "${CHECKPOINT_FILE}" ]; then
    FILE_SIZE=$(stat -c%s "${CHECKPOINT_FILE}" 2>/dev/null || echo "0")
    echo "Checkpoint file size: ${FILE_SIZE} bytes"

    # Basic validation - FCAF3D checkpoint should be around 200MB
    if [ "${FILE_SIZE}" -gt 100000000 ]; then  # 100MB
        echo "✓ Checkpoint appears valid"
    elif [ "${FILE_SIZE}" -eq 0 ]; then
        echo "⚠ Warning: Checkpoint is empty/placeholder"
    else
        echo "⚠ Warning: Checkpoint file seems small (${FILE_SIZE} bytes)"
        echo "  This might be expected if it's a placeholder file"
    fi
else
    echo "✗ Checkpoint file missing"
    exit 1
fi

# Create config file if it doesn't exist
CONFIG_FILE="${CONFIG_DIR}/fcaf3d_1x8_scannet-3d-18class.py"
if [ ! -f "${CONFIG_FILE}" ]; then
    echo "Creating FCAF3D config file..."

    # Basic FCAF3D config for ScanNet
    cat > "${CONFIG_FILE}" << 'EOF'
_base_ = [
    '../_base_/datasets/scannet-3d.py',
    '../_base_/models/fcaf3d.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]

# Model settings
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
        num_classes=18,  # ScanNet has 18 classes
        bbox_coder=dict(type='FCAF3DBBoxCoder'),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(
            type='IoU3DLoss',
            loss_weight=1.0),
        loss_dir=dict(
            type='CrossEntropyLoss',
            loss_weight=0.1),
    ),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(
        nms_pre=1000,
        nms_thr=0.5,
        score_thr=0.3,
        min_bbox_size=0,
        max_num=100)
)

# Dataset settings
data_root = './data/scannet/'
class_names = [
    'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf',
    'picture', 'counter', 'desk', 'curtain', 'refrigerator', 'showercurtrain',
    'toilet', 'sink', 'bathtub', 'garbagebin'
]

# Modify for pedestrian detection - focus on person class
# Note: ScanNet may not have explicit pedestrian labels
# This config is for demonstration; adapt for your dataset
train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        load_dim=6,
        use_dim=[0, 1, 2, 3, 4, 5]),
    dict(type='LoadAnnotations3D'),
    dict(type='GlobalAlignment', rotation_axis=2),
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.087266, 0.087266],
        scale_ratio_range=[1.0, 1.0],
        shift_height=True),
    dict(type='PointSample', num_points=40000),
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]

test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        load_dim=6,
        use_dim=[0, 1, 2, 3, 4, 5]),
    dict(type='GlobalAlignment', rotation_axis=2),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1., 1.],
                shift_height=True),
            dict(type='PointSample', num_points=40000),
            dict(
                type='Pack3DDetInputs',
                keys=['points'])
        ])
]

# Evaluator
val_evaluator = dict(type='IndoorMetric')
test_evaluator = val_evaluator

# Training settings
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=12, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# Optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.001, weight_decay=0.0001),
    clip_grad=dict(max_norm=10, norm_type=2))

# Learning rate scheduler
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.001,
        by_epoch=False,
        begin=0,
        end=1000),
    dict(
        type='MultiStepLR',
        begin=0,
        end=12,
        by_epoch=True,
        milestones=[8, 11],
        gamma=0.1)
]

# Default hooks
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1),
    logger=dict(type='LoggerHook', interval=50))

# Runtime
default_scope = 'mmdet3d'
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')
EOF

    echo "Config file created at ${CONFIG_FILE}"
else
    echo "Config file already exists at ${CONFIG_FILE}"
fi

echo ""
echo "Setup complete! FCAF3D model setup finished."
echo ""

# Check if we have a real checkpoint or just a placeholder
if [ -f "${CHECKPOINT_FILE}" ] && [ "$(stat -c%s "${CHECKPOINT_FILE}" 2>/dev/null || echo "0")" -gt 100000000 ]; then
    echo "✓ Real checkpoint downloaded - model is ready for deployment"
    STATUS="READY"
else
    echo "⚠ Placeholder checkpoint created - manual download required"
    echo "  See README.md for manual download instructions"
    STATUS="NEEDS_CHECKPOINT"
fi

echo ""
echo "Next steps:"
echo "1. Build ROCm Docker container:"
echo "   docker build -f Dockerfile.rocm -t ankk98/cvat-fcaf3d-rocm:latest ."
echo ""
echo "2. Deploy to CVAT:"
echo "   cd ../../../../  # Back to serverless root"
echo "   FCAF3D_USE_ROCM=1 ./deploy_rocm_pointcloud_models.sh fcaf3d deploy"
echo ""
echo "Model files:"
echo "- Checkpoint: ${CHECKPOINT_FILE} (${FILE_SIZE} bytes)"
echo "- Config: ${CONFIG_FILE}"
echo ""
echo "Note: This implementation uses ROCm GPU acceleration for optimal performance."
