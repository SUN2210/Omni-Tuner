# dota3c数据集（Few-Shot + DataAug）配置
# 在原始 few-shot 配置基础上叠加更强的数据增强，以缓解小样本仅 8 张图的过拟合风险。

dataset_type = 'Dota3c_fsDataset'
data_root = 'dataset/dota3c_fs/'  # 根据您提供的路径

# 定义三个类别
CLASSES = ('plane', 'storage-tank', 'ship')

# 图像归一化配置
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# 旋转等几何增强在 BGR 空间下的填充值（使用均值并转换到 BGR 顺序）
rot_fill = tuple(float(x) for x in img_norm_cfg['mean'][::-1])
cutout_ratios = [(0.08, 0.08), (0.12, 0.1), (0.18, 0.14)]

# 训练数据处理流水线（强化版）
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PhotoMetricDistortion'),
    dict(type='Expand', mean=img_norm_cfg['mean'], to_rgb=True, ratio_range=(1, 2.3), prob=0.7),
    dict(type='MinIoURandomCrop', min_ious=(0.1, 0.3, 0.5, 0.7), min_crop_size=0.4),
    dict(
        type='Resize',
        img_scale=[(640, 640), (800, 800)],
        multiscale_mode='range',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=[0.25, 0.25], direction=['horizontal', 'vertical']),
    dict(type='fs_RandomRotate', angles=[0, 90, 180, 270], prob=0.5, border_value=rot_fill),
    dict(type='CutOut', n_holes=(1, 4), cutout_ratio=cutout_ratios, fill_in=rot_fill),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

# 测试数据处理流水线（保持稳定评估）
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(896, 896),
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

# 数据集配置
data = dict(
    samples_per_gpu=4,  # 每个GPU的批次大小
    workers_per_gpu=4,   # 每个GPU的数据加载线程数
    train=dict(
        type='RepeatDataset',
        times=6,  # few-shot 下加倍重复次数以搭配增强
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

# 评估配置
evaluation = dict(
    interval=1,
    metric='mAP',
)
