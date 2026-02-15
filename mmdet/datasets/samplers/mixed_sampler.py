import math
import numpy as np
import torch
from mmcv.runner import get_dist_info
from torch.utils.data import Sampler


class MixedDatasetSampler(Sampler):
    """混合数据集采样器，用于小样本学习训练
    
    以大数据集为基准，小数据集循环使用。每个batch中固定比例地从两个数据集采样数据。
    支持RepeatDataset包装的ConcatDataset。
    
    Args:
        dataset: ConcatDataset 或 RepeatDataset(ConcatDataset)，包含两个子数据集
        samples_per_gpu (int): 每个GPU的batch size
        dataset_ratios (list): 两个数据集的采样比例，如[2, 2]表示每个batch中各采样2张
        seed (int): 随机种子
    """
    
    def __init__(self, dataset, samples_per_gpu=4, dataset_ratios=[2, 2], seed=0):
        # ===== 关键修复：处理RepeatDataset包装的ConcatDataset =====
        actual_dataset = dataset
        repeat_times = 1
        
        # 如果是RepeatDataset，获取内部的dataset
        if hasattr(dataset, 'dataset') and hasattr(dataset, 'times'):
            print(f"RepeatDataset detected, times={dataset.times}")
            actual_dataset = dataset.dataset
            repeat_times = dataset.times
        
        # 检查实际的数据集是否是ConcatDataset
        assert hasattr(actual_dataset, 'datasets'), f"需要使用ConcatDataset，但得到的是 {type(actual_dataset)}"
        assert len(actual_dataset.datasets) == 2, "目前只支持两个数据集的混合"
        assert sum(dataset_ratios) == samples_per_gpu, "采样比例之和必须等于samples_per_gpu"
        
        self.dataset = dataset  # 保持原始dataset的引用
        self.actual_dataset = actual_dataset  # ConcatDataset
        self.repeat_times = repeat_times
        self.samples_per_gpu = samples_per_gpu
        self.dataset_ratios = dataset_ratios
        self.seed = seed
        
        # 获取两个子数据集的实际长度（不考虑RepeatDataset的重复）
        self.dataset_sizes = []
        self.original_datasets = []
        
        for ds in actual_dataset.datasets:
            if hasattr(ds, 'dataset') and hasattr(ds, 'times'):
                # 这是RepeatDataset，获取原始数据集大小
                original_size = len(ds.dataset)
                repeated_size = original_size * ds.times
                self.dataset_sizes.append(repeated_size)
                self.original_datasets.append(ds.dataset)
                print(f"RepeatDataset detected: original={original_size}, times={ds.times}, total={repeated_size}")
            else:
                # 普通数据集
                self.dataset_sizes.append(len(ds))
                self.original_datasets.append(ds)
        
        self.cumulative_sizes = actual_dataset.cumulative_sizes
        
        print(f"Dataset sizes: {self.dataset_sizes}")
        print(f"Dataset ratios: {self.dataset_ratios}")
        
        # ===== 关键改进：以大数据集为基准 =====
        # 找出哪个数据集更大
        larger_dataset_idx = 0 if self.dataset_sizes[0] >= self.dataset_sizes[1] else 1
        smaller_dataset_idx = 1 - larger_dataset_idx
        
        self.larger_dataset_idx = larger_dataset_idx
        self.smaller_dataset_idx = smaller_dataset_idx
        
        # 以大数据集为基准计算batch数量
        larger_dataset_size = self.dataset_sizes[larger_dataset_idx]
        larger_dataset_ratio = self.dataset_ratios[larger_dataset_idx]
        
        # 确保能够遍历完所有大数据集的图片
        self.num_batches = math.ceil(larger_dataset_size / larger_dataset_ratio)
        self.num_samples = self.num_batches * self.samples_per_gpu
        
        print(f"Larger dataset: idx={larger_dataset_idx}, size={larger_dataset_size}")
        print(f"Smaller dataset: idx={smaller_dataset_idx}, size={self.dataset_sizes[smaller_dataset_idx]}")
        print(f"Num batches: {self.num_batches}")
        print(f"Total samples: {self.num_samples}")
        
    def _get_dataset_indices(self, dataset_idx, needed_samples):
        """获取指定数据集的索引，支持循环采样"""
        dataset_size = self.dataset_sizes[dataset_idx]
        
        # 设置随机种子
        g = torch.Generator()
        g.manual_seed(self.seed + dataset_idx)
        
        if needed_samples <= dataset_size:
            # 不需要重复采样，直接随机选择
            all_indices = torch.randperm(dataset_size, generator=g).tolist()
            selected_indices = all_indices[:needed_samples]
        else:
            # 需要循环采样
            print(f"Dataset {dataset_idx}: need {needed_samples} samples, but only has {dataset_size}, will repeat")
            full_cycles = needed_samples // dataset_size
            remaining = needed_samples % dataset_size
            
            selected_indices = []
            
            # 完整循环
            for cycle in range(full_cycles):
                cycle_indices = torch.randperm(dataset_size, generator=g).tolist()
                selected_indices.extend(cycle_indices)
            
            # 剩余部分
            if remaining > 0:
                remaining_indices = torch.randperm(dataset_size, generator=g).tolist()[:remaining]
                selected_indices.extend(remaining_indices)
        
        # 转换为全局索引（在ConcatDataset中的索引）
        if dataset_idx == 0:
            global_indices = selected_indices
        else:
            global_indices = [idx + self.cumulative_sizes[dataset_idx-1] for idx in selected_indices]
        
        return global_indices
        
    def __iter__(self):
        # 为每个数据集计算需要的样本数
        needed_samples = [self.num_batches * ratio for ratio in self.dataset_ratios]
        
        # 获取每个数据集的索引
        indices_per_dataset = []
        for i, needed in enumerate(needed_samples):
            dataset_indices = self._get_dataset_indices(i, needed)
            indices_per_dataset.append(dataset_indices)
            print(f"Dataset {i}: generated {len(dataset_indices)} indices")
        
        # 设置batch级别的随机生成器
        g = torch.Generator()
        g.manual_seed(self.seed)
        
        # 按batch组织索引 - 保持原有逻辑不变
        batch_indices = []
        debug_info = []  # 只收集前5个batch的信息
        
        for batch_idx in range(self.num_batches):
            batch = []
            
            # 从每个数据集按比例采样
            for dataset_idx, ratio in enumerate(self.dataset_ratios):
                start_idx = batch_idx * ratio
                end_idx = start_idx + ratio
                
                # 确保不越界
                dataset_indices = indices_per_dataset[dataset_idx]
                if end_idx <= len(dataset_indices):
                    current_batch_indices = dataset_indices[start_idx:end_idx]
                    batch.extend(current_batch_indices)
                else:
                    # 处理最后一个batch可能不足的情况
                    current_batch_indices = dataset_indices[start_idx:]
                    batch.extend(current_batch_indices)
            
            # ===== 收集前5个batch的调试信息（在shuffle之前）=====
            if batch_idx < 5:
                batch_debug = {
                    'batch_idx': batch_idx,
                    'before_shuffle': batch.copy(),
                    'composition': {}
                }
                # 统计组成
                for idx in batch:
                    if idx < self.cumulative_sizes[0]:
                        dataset_idx = 0
                    else:
                        dataset_idx = 1
                    batch_debug['composition'][dataset_idx] = batch_debug['composition'].get(dataset_idx, 0) + 1
                debug_info.append(batch_debug)
            
            # 打乱batch内的顺序 - 保持原有逻辑
            if len(batch) > 0:
                batch_tensor = torch.tensor(batch)
                shuffled_batch = batch_tensor[torch.randperm(len(batch), generator=g)].tolist()
                batch_indices.extend(shuffled_batch)
                
                # 记录shuffle后的结果
                if batch_idx < 5:
                    debug_info[batch_idx]['after_shuffle'] = shuffled_batch.copy()
        
        # ===== 简化的调试输出 =====
        print("\n" + "="*60)
        print("🔍 BATCH COMPOSITION - First 5 Batches")
        print("="*60)
        
        for info in debug_info:
            print(f"📦 Batch {info['batch_idx']}: {info['composition']} (target: {dict(enumerate(self.dataset_ratios))})")
        
        print("="*60)
        print(f"✅ Generated {len(batch_indices)} total samples for {self.num_batches} batches")
        print("="*60 + "\n")
        
        return iter(batch_indices)
    
    def __len__(self):
        return self.num_samples


class DistributedMixedDatasetSampler(Sampler):
    """分布式混合数据集采样器"""
    
    def __init__(self, dataset, samples_per_gpu=4, dataset_ratios=[2, 2], 
                 num_replicas=None, rank=None, seed=0):
        _rank, _num_replicas = get_dist_info()
        if num_replicas is None:
            num_replicas = _num_replicas
        if rank is None:
            rank = _rank
            
        self.dataset = dataset
        self.samples_per_gpu = samples_per_gpu
        self.dataset_ratios = dataset_ratios
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.seed = seed
        
        # 创建基础采样器
        self.base_sampler = MixedDatasetSampler(
            dataset, samples_per_gpu, dataset_ratios, seed
        )
        
        # 计算分布式参数
        total_samples = len(self.base_sampler)
        self.num_samples = math.ceil(total_samples / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas
        
        print(f"Distributed sampler: rank={rank}, num_replicas={num_replicas}")
        print(f"Base samples={total_samples}, per_replica={self.num_samples}")
        
    def __iter__(self):
        # 生成基础索引
        self.base_sampler.seed = self.epoch + self.seed
        indices = list(self.base_sampler)
        
        # 添加额外样本以便均匀分割
        if len(indices) < self.total_size:
            # 循环使用现有索引来填充
            padding_needed = self.total_size - len(indices)
            indices += (indices * ((padding_needed // len(indices)) + 1))[:padding_needed]
        
        indices = indices[:self.total_size]
        assert len(indices) == self.total_size
        
        # 分配给当前rank
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples
        
        print(f"Rank {self.rank}: got {len(indices)} samples")
        return iter(indices)
    
    def __len__(self):
        return self.num_samples
    
    def set_epoch(self, epoch):
        """设置epoch用于随机种子"""
        self.epoch = epoch
        print(f"Set epoch to {epoch} for rank {self.rank}")