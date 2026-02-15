# dota3c数据集配置
dataset_type = 'Dota3c_fsDataset'
data_root = 'dataset/dota3c_fs/'  # 根据您提供的路径

# 定义三个类别
CLASSES = ('plane', 'storage-tank', 'ship')

# 图像归一化配置
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# 训练数据处理流水线
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=[(640, 640), (800, 800)],
        multiscale_mode='range', keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

# 测试数据处理流水线
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(800, 800),
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
        times=3,  # 重复数据集3次以增加epoch的长度
        dataset=dict(
            type=dataset_type,
            ann_file=data_root + 'VOC2007/ImageSets/Main/trainval.txt',
            img_prefix=data_root + 'VOC2007/',
            pipeline=train_pipeline,
            classes=CLASSES)),  # 指定类别
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'VOC2007/ImageSets/Main/test.txt',
        img_prefix=data_root + 'VOC2007/',
        pipeline=test_pipeline,
        classes=CLASSES),  # 指定类别
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'VOC2007/ImageSets/Main/test.txt',
        img_prefix=data_root + 'VOC2007/',
        pipeline=test_pipeline,
        classes=CLASSES))  # 指定类别

# 评估配置
evaluation = dict(
    interval=1,
    metric='mAP',
)