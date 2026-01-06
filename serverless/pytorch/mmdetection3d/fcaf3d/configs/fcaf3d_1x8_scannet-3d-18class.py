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
