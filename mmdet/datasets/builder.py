import copy
import platform
import random
from functools import partial

import numpy as np
from mmcv.parallel import collate
from mmcv.runner import get_dist_info
from mmcv.utils import Registry, build_from_cfg
from torch.utils.data import DataLoader

from .samplers import (DistributedGroupSampler, DistributedSampler, GroupSampler,
                       MixedDatasetSampler, DistributedMixedDatasetSampler)


if platform.system() != 'Windows':
    # https://github.com/pytorch/pytorch/issues/973
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    hard_limit = rlimit[1]
    soft_limit = min(4096, hard_limit)
    resource.setrlimit(resource.RLIMIT_NOFILE, (soft_limit, hard_limit))

DATASETS = Registry('dataset')
PIPELINES = Registry('pipeline')


def _concat_dataset(cfg, default_args=None):
    from .dataset_wrappers import ConcatDataset
    ann_files = cfg['ann_file']
    img_prefixes = cfg.get('img_prefix', None)
    seg_prefixes = cfg.get('seg_prefix', None)
    proposal_files = cfg.get('proposal_file', None)
    separate_eval = cfg.get('separate_eval', True)

    datasets = []
    num_dset = len(ann_files)
    for i in range(num_dset):
        data_cfg = copy.deepcopy(cfg)
        # pop 'separate_eval' since it is not a valid key for common datasets.
        if 'separate_eval' in data_cfg:
            data_cfg.pop('separate_eval')
        data_cfg['ann_file'] = ann_files[i]
        if isinstance(img_prefixes, (list, tuple)):
            data_cfg['img_prefix'] = img_prefixes[i]
        if isinstance(seg_prefixes, (list, tuple)):
            data_cfg['seg_prefix'] = seg_prefixes[i]
        if isinstance(proposal_files, (list, tuple)):
            data_cfg['proposal_file'] = proposal_files[i]
        datasets.append(build_dataset(data_cfg, default_args))

    return ConcatDataset(datasets, separate_eval)


def build_dataset(cfg, default_args=None):
    from .dataset_wrappers import (ConcatDataset, RepeatDataset,
                                   ClassBalancedDataset)
    if isinstance(cfg, (list, tuple)):
        dataset = ConcatDataset([build_dataset(c, default_args) for c in cfg])
    elif cfg['type'] == 'ConcatDataset':
        dataset = ConcatDataset(
            [build_dataset(c, default_args) for c in cfg['datasets']],
            cfg.get('separate_eval', True))
    elif cfg['type'] == 'RepeatDataset':
        dataset = RepeatDataset(
            build_dataset(cfg['dataset'], default_args), cfg['times'])
    elif cfg['type'] == 'ClassBalancedDataset':
        dataset = ClassBalancedDataset(
            build_dataset(cfg['dataset'], default_args), cfg['oversample_thr'])
    elif isinstance(cfg.get('ann_file'), (list, tuple)):
        dataset = _concat_dataset(cfg, default_args)
    else:
        dataset = build_from_cfg(cfg, DATASETS, default_args)

    return dataset


def build_dataloader(dataset,
                     samples_per_gpu,
                     workers_per_gpu,
                     num_gpus=1,
                     dist=True,
                     shuffle=True,
                     seed=None,
                     sampler_cfg=None,  # 关键参数：采样器配置
                     **kwargs):
    """Build PyTorch DataLoader.

    In distributed training, each GPU/process has a dataloader.
    In non-distributed training, there is only one dataloader for all GPUs.

    Args:
        dataset (Dataset): A PyTorch dataset.
        samples_per_gpu (int): Number of training samples on each GPU, i.e.,
            batch size of each GPU.
        workers_per_gpu (int): How many subprocesses to use for data loading
            for each GPU.
        num_gpus (int): Number of GPUs. Only used in non-distributed training.
        dist (bool): Distributed training/test or not. Default: True.
        shuffle (bool): Whether to shuffle the data at every epoch.
            Default: True.
        seed (int): Random seed.
        sampler_cfg (dict): Sampler configuration. If provided, will use custom sampler.
        kwargs: any keyword argument to be used to initialize DataLoader

    Returns:
        DataLoader: A PyTorch dataloader.
    """
    rank, world_size = get_dist_info()
    
    # 处理自定义采样器配置
    if sampler_cfg is not None:
        sampler_type = sampler_cfg.get('type', None)
        print(f"Using custom sampler: {sampler_type}")
        
        if sampler_type == 'MixedDatasetSampler':
            # 混合数据集采样器
            dataset_ratios = sampler_cfg.get('dataset_ratios', [2, 2])
            
            if dist:
                # 分布式训练
                sampler = DistributedMixedDatasetSampler(
                    dataset, 
                    samples_per_gpu=samples_per_gpu,
                    dataset_ratios=dataset_ratios,
                    num_replicas=world_size,
                    rank=rank,
                    seed=seed if seed is not None else 0
                )
                batch_size = samples_per_gpu
                num_workers = workers_per_gpu
            else:
                # 非分布式训练
                sampler = MixedDatasetSampler(
                    dataset,
                    samples_per_gpu=samples_per_gpu,
                    dataset_ratios=dataset_ratios,
                    seed=seed if seed is not None else 0
                )
                batch_size = num_gpus * samples_per_gpu
                num_workers = num_gpus * workers_per_gpu
                
            print(f"MixedDatasetSampler configured:")
            print(f"  - samples_per_gpu: {samples_per_gpu}")
            print(f"  - dataset_ratios: {dataset_ratios}")
            print(f"  - distributed: {dist}")
            print(f"  - batch_size: {batch_size}")
            
        else:
            raise ValueError(f"不支持的采样器类型: {sampler_type}")
    else:
        # 使用默认采样器逻辑
        print("Using default samplers")
        if dist:
            # 分布式训练的默认采样器
            if shuffle:
                sampler = DistributedGroupSampler(
                    dataset, samples_per_gpu, world_size, rank, seed=seed)
            else:
                sampler = DistributedSampler(
                    dataset, world_size, rank, shuffle=False, seed=seed)
            batch_size = samples_per_gpu
            num_workers = workers_per_gpu
        else:
            # 非分布式训练的默认采样器
            sampler = GroupSampler(dataset, samples_per_gpu) if shuffle else None
            batch_size = num_gpus * samples_per_gpu
            num_workers = num_gpus * workers_per_gpu

    # 初始化函数
    init_fn = partial(
        worker_init_fn, num_workers=num_workers, rank=rank,
        seed=seed) if seed is not None else None

    # 创建DataLoader
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=partial(collate, samples_per_gpu=samples_per_gpu),
        pin_memory=False,
        worker_init_fn=init_fn,
        **kwargs)

    print(f"DataLoader created:")
    print(f"  - Total dataset size: {len(dataset)}")
    if sampler is not None:
        print(f"  - Sampler length: {len(sampler)}")
        print(f"  - Sampler type: {type(sampler).__name__}")
    else:
        print(f"  - Sampler: None (sequential loading)")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Num workers: {num_workers}")

    return data_loader


def worker_init_fn(worker_id, num_workers, rank, seed):
    # The seed of each worker equals to
    # num_worker * rank + worker_id + user_seed
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)