log_config = dict(
    interval=50, hooks=[dict(type='TextLoggerHook', by_epoch=False)])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = 'log/fempvit_m1_1_human_stage2/latest.pth'
workflow = [('train', 1)]
cudnn_benchmark = True
dataset_type = 'HumanDataset'
data_root = 'F:/human_seg/train_set'
train_root = 'F:/human_seg/train_set/train'
val_root = 'F:/human_seg/train_set/val'
img_norm_cfg = dict(
    mean=[114.66601625, 119.63588384, 124.47265218],
    std=[62.54581098, 62.51420716, 63.6411008],
    to_rgb=True)
resize_range = (512, 2048)
resize_ratio_range = (0.5, 2.0)
crop_size = (512, 512)
batch_size = 8
num_workers = 6
max_iters = 20000
interval = 100
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(
        type='Resize',
        img_scale=(512, 2048),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=(512, 512), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(
        type='Normalize',
        mean=[114.66601625, 119.63588384, 124.47265218],
        std=[62.54581098, 62.51420716, 63.6411008],
        to_rgb=True),
    dict(type='Pad', size=(512, 512), pad_val=144, seg_pad_val=0),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 2048),
        flip=False,
        transforms=[
            dict(type='AlignResize', keep_ratio=True, size_divisor=32),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[114.66601625, 119.63588384, 124.47265218],
                std=[62.54581098, 62.51420716, 63.6411008],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=6,
    train=dict(
        type='RepeatDataset',
        times=50,
        dataset=dict(
            type='HumanDataset',
            data_root='F:/human_seg/train_set/train',
            img_dir='images_png',
            ann_dir='masks_png_thre',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', reduce_zero_label=False),
                dict(
                    type='Resize',
                    img_scale=(512, 2048),
                    ratio_range=(0.5, 2.0),
                    keep_ratio=True),
                dict(
                    type='RandomCrop',
                    crop_size=(512, 512),
                    cat_max_ratio=0.75),
                dict(type='RandomFlip', prob=0.5),
                dict(type='PhotoMetricDistortion'),
                dict(
                    type='Normalize',
                    mean=[114.66601625, 119.63588384, 124.47265218],
                    std=[62.54581098, 62.51420716, 63.6411008],
                    to_rgb=True),
                dict(type='Pad', size=(512, 512), pad_val=144, seg_pad_val=0),
                dict(type='DefaultFormatBundle'),
                dict(type='Collect', keys=['img', 'gt_semantic_seg'])
            ])),
    val=dict(
        type='HumanDataset',
        data_root='F:/human_seg/train_set/val',
        img_dir='images_png',
        ann_dir='masks_png_thre',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(512, 2048),
                flip=False,
                transforms=[
                    dict(type='AlignResize', keep_ratio=True, size_divisor=32),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[114.66601625, 119.63588384, 124.47265218],
                        std=[62.54581098, 62.51420716, 63.6411008],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='HumanDataset',
        data_root='F:/human_seg/train_set/val',
        img_dir='images_png',
        ann_dir='masks_png_thre',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(512, 2048),
                flip=False,
                transforms=[
                    dict(type='AlignResize', keep_ratio=True, size_divisor=32),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[114.66601625, 119.63588384, 124.47265218],
                        std=[62.54581098, 62.51420716, 63.6411008],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
norm_cfg = dict(type='SyncBN', requires_grad=True)
model_variant = 'FemtoNet'
num_classes = 2
pretrain_ckpt = 'log/fempvit_m1_1_human/latest.pth'
lr = 1e-05
model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='FemtoNet',
        input_chanels=3,
        widen_factor=0.5,
        diff_conv=True,
        out_indices=(1, 2, 4, 6),
        act_cfg=dict(type='SiLU'),
        init_cfg=dict(
            type='Pretrained',
            prefix='backbone',
            checkpoint='log/fempvit_m1_1_human/latest.pth')),
    neck=dict(
        type='FPN',
        in_channels=[16, 16, 48, 160],
        out_channels=160,
        num_outs=4),
    decode_head=dict(
        type='FPNHead',
        in_channels=[160, 160, 160, 160],
        in_index=[0, 1, 2, 3],
        feature_strides=[4, 8, 16, 32],
        channels=128,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0,
            class_weight=[0.5, 2.0])),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
gpu_multiples = 1
optimizer = dict(type='AdamW', lr=1e-05, weight_decay=0.0001)
optimizer_config = dict()
lr_config = dict(policy='poly', power=0.9, min_lr=1e-06, by_epoch=False)
runner = dict(type='IterBasedRunner', max_iters=20000)
checkpoint_config = dict(by_epoch=False, interval=100)
evaluation = dict(interval=100, metric='mIoU')
work_dir = 'log/fempvit_m1_1_human_stage2'
gpu_ids = [device(type='cuda', index=0)]
