# 小样本学习混合数据集配置
# 训练时：batch_size=4，其中2张来自xview3c，2张来自dota3c_fs
# 验证时：全部使用dota3c_fs验证集

# 数据集类型
xview3c_dataset_type = 'Xview3cDataset'
dota3c_fs_dataset_type = 'Dota3c_fsDataset'

# 数据路径
xview3c_data_root = 'dataset/xview3c/'
dota3c_fs_data_root = 'dataset/dota3c_fs/'

# 类别定义（三类目标检测）
CLASSES = ('plane', 'storage-tank', 'ship')

# 图像归一化配置
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], 
    std=[58.395, 57.12, 57.375], 
    to_rgb=True
)

# 训练数据处理流水线
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(800,800), keep_ratio=True),
    # dict(type='Resize', img_scale=[(660, 660), (800, 880)],
    #     multiscale_mode='range', keep_ratio=True),
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

# XVIEW3C训练数据集配置
xview3c_train_dataset = dict(
    type=xview3c_dataset_type,
    ann_file=xview3c_data_root + 'VOC2007/ImageSets/Main/trainval.txt',
    img_prefix=xview3c_data_root + 'VOC2007/',
    pipeline=train_pipeline,
    classes=CLASSES
)

# DOTA3C Few-Shot训练数据集配置
dota3c_fs_train_dataset = dict(
    type=dota3c_fs_dataset_type,
    ann_file=dota3c_fs_data_root + 'VOC2007/ImageSets/Main/trainval.txt',
    img_prefix=dota3c_fs_data_root + 'VOC2007/',
    pipeline=train_pipeline,
    classes=CLASSES
)

# 混合训练数据集配置
mixed_train_dataset = dict(
    type='ConcatDataset',
    datasets=[xview3c_train_dataset, dota3c_fs_train_dataset],
    separate_eval=True  # 分别评估
)

# DOTA3C Few-Shot验证数据集配置（只用这一个进行验证）
dota3c_fs_val_dataset = dict(
    type=dota3c_fs_dataset_type,
    ann_file=dota3c_fs_data_root + 'VOC2007/ImageSets/Main/test.txt',
    img_prefix=dota3c_fs_data_root + 'VOC2007/',
    pipeline=test_pipeline,
    classes=CLASSES
)

# 混合采样器配置
mixed_sampler_cfg = dict(
    type='MixedDatasetSampler',
    dataset_ratios=[2, 2]  # xview3c: 2张, dota3c_fs: 2张
)

# 数据配置
data = dict(
    samples_per_gpu=4,  # 每个GPU的批次大小为4
    workers_per_gpu=2,   # 每个GPU的数据加载线程数
    train=dict(
        type='RepeatDataset',
        times=1,  # 重复数据集以增加训练轮数
        dataset=mixed_train_dataset
    ),
    val=dota3c_fs_val_dataset,  # 验证只使用DOTA3C Few-Shot数据集
    test=dota3c_fs_val_dataset,  # 测试也使用DOTA3C Few-Shot数据集
    # 自定义采样器配置
    sampler_cfg=mixed_sampler_cfg
)

# 评估配置
evaluation = dict(
    interval=1,
    metric='mAP',
    # save_best='auto'  # 自动保存最佳模型
)

# 数据加载器配置提示
# 在训练脚本中需要传递sampler_cfg参数给build_dataloader函数
# 例如：
# train_dataloader = build_dataloader(
#     train_dataset,
#     cfg.data.samples_per_gpu,
#     cfg.data.workers_per_gpu,
#     len(gpus),
#     dist=distributed,
#     seed=cfg.seed,
#     sampler_cfg=cfg.data.get('sampler_cfg', None)
# )