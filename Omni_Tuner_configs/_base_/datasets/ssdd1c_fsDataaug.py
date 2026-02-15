# ssdd1c数据集（Few-Shot + DataAug）配置
# 为 8 张左右的小样本场景补充多样性增强，提升模型泛化能力。

dataset_type = 'Ssdd1c_fsDataset'
data_root = 'dataset/ssdd1c_fs/'  # 根据您提供的路径

# 定义类别
CLASSES = ('ship',)

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

rot_fill = tuple(float(x) for x in img_norm_cfg['mean'][::-1])
cutout_ratios = [(0.08, 0.08), (0.12, 0.1), (0.18, 0.14)]

train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PhotoMetricDistortion'),
    dict(type='Expand', mean=img_norm_cfg['mean'], to_rgb=True, ratio_range=(1, 2.3), prob=0.7),
    dict(type='MinIoURandomCrop', min_ious=(0.1, 0.3, 0.5, 0.7), min_crop_size=0.4),
    dict(type='Resize', img_scale=(1000, 600), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=[0.25, 0.25], direction=['horizontal', 'vertical']),
    dict(type='fs_RandomRotate', angles=[0, 90, 180, 270], prob=0.5, border_value=rot_fill),
    dict(type='CutOut', n_holes=(1, 4), cutout_ratio=cutout_ratios, fill_in=rot_fill),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1000, 600),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type='RepeatDataset',
        times=6,
        dataset=dict(
            type=dataset_type,
            ann_file=data_root + 'VOC2007/ImageSets/Main/trainval.txt',
            img_prefix=data_root + 'VOC2007/',
            pipeline=train_pipeline,
            classes=CLASSES)),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'VOC2007/ImageSets/Main/test.txt',
        img_prefix=data_root + 'VOC2007/',
        pipeline=test_pipeline,
        classes=CLASSES),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'VOC2007/ImageSets/Main/test.txt',
        img_prefix=data_root + 'VOC2007/',
        pipeline=test_pipeline,
        classes=CLASSES))

evaluation = dict(
    interval=1,
    metric='mAP',
)
