_base_ = [
    '../_base_/default_runtime.py'
]

# ===================== Dataset =====================
dataset_type = 'HumanDataset'
data_root = 'F:/human_seg/train_set'
train_root = f'{data_root}/train'
val_root = f'{data_root}/val'
img_norm_cfg = dict(
    mean=[114.66601625, 119.63588384, 124.47265218],
    std=[62.54581098, 62.51420716, 63.6411008],
    to_rgb=True
)
resize_range = (512, 2048)
resize_ratio_range = (0.5, 2.0)
crop_size = (512, 512)
batch_size = 8
num_workers = 6
max_iters = 20000
interval = 500

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='Resize', img_scale=resize_range, ratio_range=resize_ratio_range, keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=144, seg_pad_val=0),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=resize_range,
        flip=False,
        transforms=[
            dict(type='AlignResize', keep_ratio=True, size_divisor=32),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=batch_size,
    workers_per_gpu=num_workers,
    train=dict(
        type='RepeatDataset',
        times=50,
        dataset=dict(
            type=dataset_type,
            data_root=train_root,
            img_dir='images_png',
            ann_dir='masks_png_thre',
            pipeline=train_pipeline)),
    val=dict(
        type=dataset_type,
        data_root=val_root,
        img_dir='images_png',
        ann_dir='masks_png_thre',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=val_root,
        img_dir='images_png',
        ann_dir='masks_png_thre',
        pipeline=test_pipeline)
)

# ===================== Model =====================
norm_cfg = dict(type='SyncBN', requires_grad=True)
model_variant = 'repvit_m1_1'
pretrain_ckpt = None
pretrain_ckpt = 'log/fpn_repvit_m1_1_human_stage1/latest.pth'
num_classes = 2
# stage1
# pretrain_ckpt = 'log/repvit_m1_1_human/latest.pth'
# lr = 0.00005
# stage 2
# pretrain_ckpt = 'log/fpn_repvit_m1_1_human_stage2/latest.pth'
lr = 0.000025

model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type=model_variant,
        style='pytorch',
        init_cfg=dict(
            type='Pretrained',
            prefix='backbone',
            checkpoint=pretrain_ckpt,
        ),
        out_indices=[3, 7, 21, 24]
    ),
    neck=dict(
        type='FPN',
        in_channels=[64, 128, 256, 512],
        out_channels=256,
        num_outs=4
    ),
    decode_head=dict(
        type='FPNHead',
        in_channels=[256, 256, 256, 256],
        in_index=[0, 1, 2, 3],
        feature_strides=[4, 8, 16, 32],
        channels=128,
        dropout_ratio=0.1,
        num_classes=num_classes,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0,
            class_weight=[0.5, 2.0]  # background, human
        )
    ),
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)

# ===================== Optimizer & Schedule =====================
gpu_multiples = 1
optimizer = dict(type='AdamW', lr=lr * gpu_multiples, weight_decay=0.0001)
optimizer_config = dict()

lr_config = dict(policy='poly', power=0.9, min_lr=1e-6, by_epoch=False)

runner = dict(type='IterBasedRunner', max_iters=max_iters // gpu_multiples)
checkpoint_config = dict(by_epoch=False, interval=interval // gpu_multiples)
evaluation = dict(interval=interval // gpu_multiples, metric='mIoU')