_base_ = [
    '../_base_/models/fpn_r50.py',
    '../_base_/datasets/human.py',
    '../_base_/default_runtime.py'
]

data_path = 'F:/human_seg/train_set'
data_size = 512
num_worker = 8
model_variant = 'repvit_m1_1'
batch_size = 256
train_epochs = 300
num_class = 1
interval = 100
pretrain_ckpt = 'segmentation/pretrain/repvit_m1_1_ade20k.pth'

# model settings
model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type=model_variant,
        style='pytorch',
        init_cfg=dict(
            type='Pretrained',
            checkpoint=pretrain_ckpt,
        ),
        out_indices = [3,7,21,24]
    ),
    neck=dict(in_channels=[64, 128, 256, 512]),
    decode_head=dict(num_classes=num_class))

gpu_multiples = 1  # we use 8 gpu instead of 4 in mmsegmentation, so lr*2 and max_iters/2
# optimizer
optimizer = dict(type='AdamW', lr=0.0001 * gpu_multiples, weight_decay=0.0001)
optimizer_config = dict()
# learning policy
lr_config = dict(policy='poly', power=0.9, min_lr=1e-6, by_epoch=False)
# runtime settings
runner = dict(type='IterBasedRunner', max_iters=10000 // gpu_multiples)
checkpoint_config = dict(by_epoch=False, interval=interval // gpu_multiples)
evaluation = dict(interval=interval // gpu_multiples, metric='mIoU')
