# dataset settings
dataset_type = 'HumanDataset'
data_root = 'F:/human_seg/train_set'
train_root = 'F:/human_seg/train_set/train'
val_root = 'F:/human_seg/train_set/val'
img_norm_cfg = dict(
    mean=[114.66601625, 119.63588384, 124.47265218], std=[62.54581098, 62.51420716, 63.6411008], to_rgb=True)
crop_size = (512, 512)
batch_size = 4
num_workers = 4

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='Resize', img_scale=(2048, 512), ratio_range=(0.5, 2.0), keep_ratio=True),
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
        img_scale=(2048, 512),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
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
        pipeline=test_pipeline))
