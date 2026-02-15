#!/usr/bin/env python3
"""
实例级原型提取

提取跨域适配所需的类别原型。

使用需要conda环境名为mmcv14

流程概览：先根据配置加载 mmdetection 检测器与权重，遍历源域数据集获取每个 GT 框的 ROI 特征，随后按类别聚合特征并调用 Sinkhorn 原型学习算法生成类别原型，最终将原型与元信息写入磁盘。

核心组件：
- :func:`_extract_roi_features` 统一处理两阶段与单阶段检测器的 ROI 特征提取；
- :func:`compute_prototypes_sinkhorn` 与 :class:`PrototypeLearnerSimple` 复现 Sinkhorn 原型学习流程；
- :func:`extract_prototypes` 负责串联数据加载、特征缓存、聚类与结果保存的主流程。

This script loads a mmdetection config and checkpoint, runs the backbone/ROI
feature extractor over the source domain training set, and clusters pooled ROI
features into per-class prototypes using a Sinkhorn-based learner. The aggregated
statistics are saved as a PyTorch checkpoint that can be reused during
target-domain adaptation.
"""
import argparse
from collections import defaultdict
import os
import sys
import time
from copy import deepcopy
from typing import Dict, List, Tuple
import sys  # ensure local mmdet modules are found
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, '.')

import mmcv
import torch
import torch.nn.functional as F
from mmcv import Config, DictAction
from mmcv.parallel import DataContainer
from mmcv.runner import load_checkpoint
from mmcv.utils import ProgressBar
from torch.utils.data import DataLoader, TensorDataset

from mmdet.core import bbox2roi
from mmdet.datasets import build_dataset, build_dataloader
from mmdet.models import build_detector
from mmdet.models.builder import build_roi_extractor


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, '.')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='从已训练的源域检测器中提取ROI特征。'
    )
    parser.add_argument(
        '--config',
        default='./work_dirs/Omni_Tuner_configs/swin-l_xview3c/xview3c_retinanet_swin_large_1x_myvpt2.py',
        help='mmdet训练用的配置文件路径。')
    parser.add_argument(
        '--checkpoint',
        default='./work_dirs/work_dirs/xview3c_retinanet_swin_large_1x_myvpt2/best.pth',
        help='已训练模型权重(.pth)文件路径。')
    parser.add_argument(
        '--out',
        default='./work_dirs/work_dirs/xview3c_retinanet_swin_large_1x_myvpt2/xview3c_prototypes_screen.pth',
        help='原型统计信息输出路径(.pth)。')
    parser.add_argument('--device', default='cuda:1', help='用于特征提取的设备（如cuda:0或cpu）。')
    parser.add_argument('--batch-size', type=int, default=16, help='特征提取时每GPU的样本数。')
    parser.add_argument('--workers', type=int, default=8, help='数据加载的工作线程数。')
    parser.add_argument('--max-samples', type=int, default=None, help='最多处理的图片数量（用于快速测试）。')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction, help='用于覆盖配置文件中的设置。')
    parser.add_argument('--no-progress', action='store_true', help='关闭进度条显示。')
    parser.add_argument('--l2-normalize', action='store_true', help='为每个原型额外保存L2归一化副本。')
    parser.add_argument('--num-prototypes', type=int, default=3, help='每类保留的原型数量。')
    parser.add_argument('--normalize-features', dest='normalize_features', action='store_true', help='聚类前对ROI特征做L2归一化（默认开启）。')
    parser.add_argument('--no-normalize-features', dest='normalize_features', action='store_false', help='聚类前不做特征归一化。')
    parser.set_defaults(normalize_features=True)
    parser.add_argument('--sinkhorn-epochs', type=int, default=100, help='Sinkhorn原型学习的训练轮数。')
    parser.add_argument('--sinkhorn-batch-size', type=int, default=512, help='Sinkhorn原型学习的批量大小。')
    parser.add_argument('--sinkhorn-queue-size', type=int, default=8192, help='Sinkhorn原型学习的队列长度。')
    parser.add_argument('--sinkhorn-momentum', type=float, default=0.02, help='Sinkhorn原型更新的动量。')
    parser.add_argument('--sinkhorn-iterations', type=int, default=5, help='每步Sinkhorn迭代次数。')
    parser.add_argument('--sinkhorn-epsilon', type=float, default=1e-2, help='Sinkhorn迭代的熵正则权重。')
    parser.add_argument('--save-tokens', default='work_dirs/xview3c_retinanet_swin_large_1x_myvpt2/xview3c_tokens_screen.pth', help='可选，保存原始实例特征的路径，便于后处理。')
    parser.add_argument('--no-visualize', action='store_true', help='关闭可视化生成（tokens和原型）。')
    parser.add_argument('--visual-max-tokens', type=int, default=5000, help='2D可视化时最多采样的token数量。')
    parser.add_argument('--visual-alpha', type=float, default=0.55, help='token散点图的透明度。')
    parser.add_argument('--visual-seed', type=int, default=0, help='可视化采样的随机种子。')
    
    # 新增特征过滤相关参数（默认开启）
    parser.add_argument('--enable-filtering', dest='enable_filtering', action='store_true', help='启用特征过滤以去除异常样本（默认开启）。')
    parser.add_argument('--disable-filtering', dest='enable_filtering', action='store_false', help='禁用特征过滤。')
    parser.set_defaults(enable_filtering=True)
    parser.add_argument('--filter-outliers-ratio', type=float, default=0.1, help='离群点过滤比例')
    parser.add_argument('--filter-knn-ratio', type=float, default=0.05, help='KNN密度过滤比例')
    parser.add_argument('--filter-activation-ratio', type=float, default=0.05, help='激活强度过滤比例')
    
    return parser.parse_args()


def _maybe_import_custom_modules(cfg: Config) -> None:
    custom_imports_cfg = cfg.get('custom_imports')
    if not custom_imports_cfg:
        return
    from mmcv.utils import import_modules_from_strings

    import_modules_from_strings(**custom_imports_cfg)


def _unwrap_data_container(container):
    if isinstance(container, DataContainer):
        data = container.data
        if isinstance(data, (list, tuple)):
            # DataContainer stores a list per GPU. When stack=False the first
            # element is itself the per-sample list we need, otherwise keep the
            # full list so downstream logic can iterate over the batch.
            if len(data) == 1 and isinstance(data[0], (list, tuple)):
                return list(data[0])
            if getattr(container, 'stack', False):
                if len(data) == 1:
                    return data[0]
                return torch.stack([item for item in data])
            return list(data)
        return data
    return container


def _ensure_single_dataset(train_cfg: Dict) -> Dict:
    """Return the underlying dataset config even if RepeatDataset is used."""
    cfg = deepcopy(train_cfg)
    if isinstance(cfg, dict) and cfg.get('type') == 'RepeatDataset':
        cfg = cfg['dataset']
    return cfg


_SINGLE_STAGE_ROI_EXTRACTOR_CACHE: Dict[Tuple[Tuple[int, ...], Tuple[int, ...]], torch.nn.Module] = {}


def _to_stride(value) -> int:
    if torch.is_tensor(value):
        return int(value.item())
    if isinstance(value, (list, tuple)):
        if not value:
            raise ValueError('Empty stride tuple encountered when building ROI extractor')
        return _to_stride(value[0])
    return int(value)


def _get_single_stage_roi_extractor(model, feats, device):
    if not hasattr(model, 'bbox_head'):
        raise AttributeError('Single-stage detector missing `bbox_head`, cannot extract ROI features.')

    featmap_strides = getattr(model.bbox_head, 'featmap_strides', None)
    if featmap_strides is None:
        anchor_generator = getattr(model.bbox_head, 'anchor_generator', None)
        if anchor_generator is not None:
            featmap_strides = getattr(anchor_generator, 'strides', None)
    if featmap_strides is None:
        raise AttributeError('Unable to infer feature map strides for single-stage detector.')

    featmap_strides = tuple(_to_stride(s) for s in featmap_strides)
    channel_signature = tuple(int(feat.shape[1]) for feat in feats[:len(featmap_strides)])
    cache_key = (featmap_strides, channel_signature)

    extractor = _SINGLE_STAGE_ROI_EXTRACTOR_CACHE.get(cache_key)
    if extractor is None:
        out_channels = channel_signature[0]
        extractor_cfg = dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=out_channels,
            featmap_strides=list(featmap_strides),
        )
        extractor = build_roi_extractor(extractor_cfg)
        extractor = extractor.to(device)
        extractor.eval()
        _SINGLE_STAGE_ROI_EXTRACTOR_CACHE[cache_key] = extractor
    return extractor


def _extract_roi_features(model, feats, rois, device):
    if hasattr(model, 'roi_head') and hasattr(model.roi_head, 'bbox_roi_extractor'):
        bbox_roi_extractor = model.roi_head.bbox_roi_extractor
        roi_feats = bbox_roi_extractor(
            feats[:bbox_roi_extractor.num_inputs], rois)
        if getattr(model.roi_head, 'with_shared_head', False):
            roi_feats = model.roi_head.shared_head(roi_feats)
        return roi_feats

    extractor = _get_single_stage_roi_extractor(model, feats, device)
    roi_feats = extractor(feats[:extractor.num_inputs], rois)
    return roi_feats


def _compute_assignments(feats: torch.Tensor, centers: torch.Tensor, normalize: bool) -> torch.Tensor:
    if feats.numel() == 0:
        return torch.empty(0, dtype=torch.long)
    if normalize:
        feats = F.normalize(feats, dim=1)
        centers = F.normalize(centers, dim=1)
        similarities = feats @ centers.t()
        return similarities.argmax(dim=1)
    distances = torch.cdist(feats, centers)
    return distances.argmin(dim=1)


# ======================================================================
# 关键过滤代码
# ======================================================================

def filter_outliers_by_distance(features: torch.Tensor, filter_ratio: float = 0.1) -> torch.Tensor:
    """过滤2：基于到类中心的距离进行离群点过滤。
    
    它计算类内所有特征的质心，然后剔除离这个质心最远的 `filter_ratio` 比例的样本。
    """
    if features.size(0) <= 1:
        return torch.ones(features.size(0), dtype=torch.bool)
    
    # 计算类中心
    center = features.mean(dim=0, keepdim=True)
    # 计算每个样本到中心的欧式距离
    distances = torch.cdist(features, center).squeeze()
    # 计算要保留的数量
    k_keep = int(features.size(0) * (1 - filter_ratio))
    k_keep = max(1, k_keep)  # 至少保留1个
    # 获取距离最小的k个样本的索引
    _, indices = distances.topk(k_keep, largest=False)
    # 创建mask，只保留这些索引对应的样本
    mask = torch.zeros(features.size(0), dtype=torch.bool)
    mask[indices] = True
    return mask


def filter_by_knn_density(features: torch.Tensor, filter_ratio: float = 0.05, k: int = 5) -> torch.Tensor:
    """过滤3：基于KNN（K近邻）的密度进行过滤。
    
    它计算每个样本与其K个最近邻居的平均距离，这个值反比于局部密度。
    然后剔除局部密度最低（即平均距离最大）的 `filter_knn_ratio` 比例的样本。
    """
    if features.size(0) <= k:
        return torch.ones(features.size(0), dtype=torch.bool)
    
    # 计算两两样本间的距离矩阵
    dist_matrix = torch.cdist(features, features)
    # 忽略自己到自己的距离
    dist_matrix.fill_diagonal_(float('inf'))
    # 找到每个样本的k个最近邻的距离
    knn_distances, _ = dist_matrix.topk(min(k, features.size(0)-1), largest=False, dim=1)
    # 计算平均KNN距离（作为密度的反向指标）
    avg_knn_dist = knn_distances.mean(dim=1)
    # 保留密度高的样本（即平均距离小的）
    k_keep = int(features.size(0) * (1 - filter_ratio))
    k_keep = max(1, k_keep)
    _, indices = avg_knn_dist.topk(k_keep, largest=False)
    mask = torch.zeros(features.size(0), dtype=torch.bool)
    mask[indices] = True
    return mask


def filter_by_activation_strength(features: torch.Tensor, filter_ratio: float = 0.05) -> torch.Tensor:
    """过滤1：基于特征激活强度（L2范数）进行过滤。
    
    它计算每个特征向量的L2范数，并剔除范数最低的 `filter_activation_ratio` 比例的样本。
    这旨在移除模型响应最弱的样本。
    """
    if features.size(0) <= 1:
        return torch.ones(features.size(0), dtype=torch.bool)
    
    # 计算每个特征的L2范数（激活强度）
    activation_strength = features.norm(dim=1)
    # 保留激活强度高的样本
    k_keep = int(features.size(0) * (1 - filter_ratio))
    k_keep = max(1, k_keep)
    _, indices = activation_strength.topk(k_keep, largest=True)
    mask = torch.zeros(features.size(0), dtype=torch.bool)
    mask[indices] = True
    return mask


def apply_feature_filtering(
    features_per_class: Dict[int, torch.Tensor],
    filter_outliers_ratio: float = 0.1,
    filter_knn_ratio: float = 0.05,
    filter_activation_ratio: float = 0.05,
    verbose: bool = True
) -> Dict[int, torch.Tensor]:
    """组合应用多种过滤方法。
    
    这是一个调度函数，按顺序应用上面定义的三个过滤器：
    1. 激活强度过滤
    2. 离群点过滤
    3. KNN密度过滤
    """
    filtered_features = {}
    total_before = 0
    total_after = 0
    
    for cls_id, features in features_per_class.items():
        original_count = features.size(0)
        total_before += original_count
        
        if original_count <= 5:  # 样本太少则不过滤
            filtered_features[cls_id] = features
            total_after += original_count
            if verbose:
                print(f"Class {cls_id}: {original_count} samples (too few to filter)")
            continue
            
        # 关键过滤流程：依次应用三种过滤策略
        # 过滤1：基于激活强度
        mask1 = filter_by_activation_strength(features, filter_activation_ratio)
        features_after_activation = features[mask1]
        
        # 过滤2：基于离群点距离
        mask2 = filter_outliers_by_distance(features_after_activation, filter_outliers_ratio)
        features_after_outlier = features_after_activation[mask2]
        
        # 过滤3：基于KNN密度
        mask3 = filter_by_knn_density(features_after_outlier, filter_knn_ratio)
        features_final = features_after_outlier[mask3]
        
        filtered_features[cls_id] = features_final
        total_after += features_final.size(0)
        
        if verbose:
            final_count = features_final.size(0)
            print(f"Class {cls_id}: {original_count} -> {final_count} "
                  f"(kept {final_count/original_count*100:.1f}%)")
    
    if verbose:
        print(f"\nTotal: {total_before} -> {total_after} "
              f"(kept {total_after/total_before*100:.1f}%)")
    
    return filtered_features

# ======================================================================
# 过滤代码注释结束
# ======================================================================

def mask_sinkhorn(
    C: torch.Tensor,
    mask: torch.Tensor,
    row_group_size: torch.Tensor,
    col_group_size: torch.Tensor,
    iterations: int = 10,
    epsilon: float = 1e-2,
) -> torch.Tensor:
    mask = mask.bool()
    row_group_size = torch.clamp(row_group_size, min=1).float()
    col_group_size = torch.clamp(col_group_size, min=1).float()

    a = torch.ones(C.shape[0], device=C.device, dtype=C.dtype) / row_group_size
    b = torch.ones(C.shape[1], device=C.device, dtype=C.dtype) / col_group_size

    u = torch.zeros_like(a)
    v = torch.zeros_like(b)
    inf_mask = torch.zeros_like(C)
    inf_mask[~mask] = -torch.inf

    def _log_boltzmann_kernel(u_vec, v_vec, cost):
        kernel = (-cost + u_vec.unsqueeze(1) + v_vec.unsqueeze(0)) / epsilon
        return kernel + inf_mask

    for _ in range(iterations):
        K = _log_boltzmann_kernel(u, v, C)
        u_update = torch.log(a + 1e-8) - torch.logsumexp(K, dim=1)
        u = epsilon * torch.nan_to_num(u_update, nan=0.0, posinf=0.0, neginf=0.0) + u

        K_t = _log_boltzmann_kernel(u, v, C).transpose(-2, -1)
        v_update = torch.log(b + 1e-8) - torch.logsumexp(K_t, dim=1)
        v = epsilon * torch.nan_to_num(v_update, nan=0.0, posinf=0.0, neginf=0.0) + v

    K = _log_boltzmann_kernel(u, v, C)
    pi = torch.exp(K)
    pi = pi * mask.float()
    return pi


class PrototypeLearnerSimple:
    def __init__(
        self,
        num_classes: int,
        num_prototypes: int,
        embed_dim: int,
        queue_size: int,
        momentum: float,
        iterations: int,
        epsilon: float,
        normalize: bool,
        device: torch.device,
    ) -> None:
        self.num_classes = num_classes
        self.num_prototypes = num_prototypes
        self.embed_dim = embed_dim
        self.queue_size = queue_size
        self.momentum = momentum
        self.iterations = iterations
        self.epsilon = epsilon
        self.normalize = normalize
        self.device = torch.device(device)

        prototypes = torch.randn(num_classes * num_prototypes, embed_dim)
        if self.normalize:
            prototypes = F.normalize(prototypes, dim=-1)
        self.prototypes = prototypes.to(self.device)

        self.class_ids_per_proto = torch.arange(num_classes, device=self.device).repeat_interleave(num_prototypes)
        self.row_group_size = (self.class_ids_per_proto[:, None] == self.class_ids_per_proto[None, :]).sum(1).float()
        self.queue_tokens = torch.empty(0, embed_dim, device=self.device)
        self.queue_labels = torch.empty(0, dtype=torch.long, device=self.device)

    def update(self, tokens: torch.Tensor, labels: torch.Tensor) -> None:
        if tokens.numel() == 0:
            return
        tokens = tokens.to(self.device).float()
        labels = labels.to(self.device).long()
        if self.normalize:
            tokens = F.normalize(tokens, dim=-1)

        self.queue_tokens = torch.cat([tokens, self.queue_tokens], dim=0)
        self.queue_labels = torch.cat([labels, self.queue_labels], dim=0)

        if self.queue_tokens.size(0) > self.queue_size:
            self.queue_tokens = self.queue_tokens[:self.queue_size]
            self.queue_labels = self.queue_labels[:self.queue_size]

        q_tokens = self.queue_tokens
        q_labels = self.queue_labels
        mask = self.class_ids_per_proto[:, None] == q_labels[None, :]
        if not mask.any():
            return

        label_group_size = (q_labels[:, None] == q_labels[None, :]).sum(1).float().clamp_min_(1.0)
        C = -(self.prototypes @ q_tokens.t())
        pi = mask_sinkhorn(
            C,
            mask,
            self.row_group_size,
            label_group_size,
            iterations=self.iterations,
            epsilon=self.epsilon,
        )
        denom = pi.sum(0, keepdim=True).clamp_min(1e-6)
        new_prototypes = (pi / denom) @ q_tokens
        self.prototypes = (1 - self.momentum) * self.prototypes + self.momentum * new_prototypes
        if self.normalize:
            self.prototypes = F.normalize(self.prototypes, dim=-1)

    def predict(self, tokens: torch.Tensor) -> torch.Tensor:
        tokens = tokens.to(self.device).float()
        if self.normalize:
            tokens = F.normalize(tokens, dim=-1)
        prototypes = self.prototypes.view(self.num_classes, self.num_prototypes, self.embed_dim)
        scores = torch.matmul(tokens, prototypes.flatten(0, 1).t())
        scores = scores.view(tokens.size(0), self.num_classes, self.num_prototypes).max(dim=-1).values
        labels = scores.argmax(dim=-1)
        return labels.cpu()

    def get_prototypes(self) -> torch.Tensor:
        return self.prototypes.detach().cpu()


def compute_prototypes_sinkhorn(
    features_per_class: Dict[int, torch.Tensor],
    num_prototypes: int,
    normalize_features: bool,
    device: torch.device,
    epochs: int,
    batch_size: int,
    queue_size: int,
    momentum: float,
    iterations: int,
    epsilon: float,
) -> Tuple[Dict[int, Dict[str, torch.Tensor]], Dict[str, float]]:
    if not features_per_class:
        raise ValueError('features_per_class is empty, cannot run sinkhorn clustering')

    class_ids = sorted(features_per_class.keys())
    class_to_index = {cls_id: idx for idx, cls_id in enumerate(class_ids)}
    embed_dim = next(iter(features_per_class.values())).size(1)

    tokens_list: List[torch.Tensor] = []
    labels_list: List[torch.Tensor] = []
    for cls_id in class_ids:
        feats = features_per_class[cls_id].float()
        tokens_list.append(feats)
        labels_list.append(torch.full((feats.size(0),), class_to_index[cls_id], dtype=torch.long))

    all_tokens = torch.cat(tokens_list, dim=0)
    all_labels = torch.cat(labels_list, dim=0)

    dataset = TensorDataset(all_tokens, all_labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    total_batches = len(loader)

    learner = PrototypeLearnerSimple(
        num_classes=len(class_ids),
        num_prototypes=num_prototypes,
        embed_dim=embed_dim,
        queue_size=queue_size,
        momentum=momentum,
        iterations=iterations,
        epsilon=epsilon,
        normalize=normalize_features,
        device=device,
    )

    print(
        f"[sinkhorn] start training: epochs={epochs}, batches_per_epoch={total_batches}, "
        f"tokens_per_epoch={dataset.tensors[0].size(0)}"
    )

    for epoch in range(epochs):
        epoch_start = time.time()
        seen_tokens = 0
        for token_batch, label_batch in loader:
            learner.update(token_batch, label_batch)
            seen_tokens += token_batch.size(0)
        epoch_time = time.time() - epoch_start
        print(
            f"[sinkhorn] epoch {epoch + 1}/{epochs}: processed {seen_tokens} tokens "
            f"in {epoch_time:.2f}s"
        )

    proto_tensor = learner.get_prototypes().view(len(class_ids), num_prototypes, embed_dim)

    with torch.no_grad():
        preds = []
        eval_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        for token_batch, _ in eval_loader:
            preds.append(learner.predict(token_batch))
    pred_labels = torch.cat(preds)
    overall_acc = (pred_labels == all_labels).float().mean().item()

    per_class_acc = {}
    for cls_id in class_ids:
        mask = all_labels == class_to_index[cls_id]
        if mask.any():
            per_class_acc[int(cls_id)] = (pred_labels[mask] == all_labels[mask]).float().mean().item()

    prototypes: Dict[int, Dict[str, torch.Tensor]] = {}
    for idx, cls_id in enumerate(class_ids):
        centers = proto_tensor[idx]
        feats = features_per_class[cls_id]
        assignments = _compute_assignments(feats, centers, normalize_features)
        counts = torch.bincount(assignments, minlength=num_prototypes).to(torch.int64)
        prototypes[int(cls_id)] = {
            'centers': centers.cpu(),
            'counts': counts.cpu(),
        }

    metrics = {'overall_accuracy': overall_acc, 'per_class_accuracy': per_class_acc}
    return prototypes, metrics


def _pca_project(feature_tensor: torch.Tensor, k: int = 2) -> torch.Tensor:
    """Lightweight PCA using SVD (center then project). Returns (N,k)."""
    if feature_tensor.size(1) <= k:
        return feature_tensor[:, :k]
    X = feature_tensor - feature_tensor.mean(0, keepdim=True)
    # Use torch.linalg.svd for stability
    try:
        U, S, Vh = torch.linalg.svd(X, full_matrices=False)
    except RuntimeError:
        # fallback to CPU
        X_cpu = X.cpu()
        U, S, Vh = torch.linalg.svd(X_cpu, full_matrices=False)
        U = U.to(feature_tensor.device)
        Vh = Vh.to(feature_tensor.device)
    comps = Vh[:k]
    return (X @ comps.T)


def _tsne_project(feature_tensor: torch.Tensor, k: int = 2, seed: int = 0) -> torch.Tensor:
    """t-SNE projection using scikit-learn. Returns (N, k)."""
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        print('[visualize] scikit-learn not available, skipping t-SNE.')
        return torch.zeros(feature_tensor.size(0), k)

    X = feature_tensor.cpu().numpy()
    n_samples = X.shape[0]
    # Perplexity must be less than the number of samples
    perplexity = min(30.0, float(n_samples - 1))
    if perplexity <= 1.0:
        print(f'[visualize] Not enough samples ({n_samples}) for t-SNE, skipping.')
        return torch.zeros(feature_tensor.size(0), k)

    tsne = TSNE(n_components=k, perplexity=perplexity, n_iter=300, random_state=seed, init='pca', learning_rate='auto')
    proj = tsne.fit_transform(X)
    return torch.from_numpy(proj)


def _visualize_tokens_and_prototypes(
    out_dir: str,
    stacked_features: Dict[int, torch.Tensor],
    prototypes: Dict[int, Dict[str, torch.Tensor]],
    metadata: Dict,
    max_tokens: int,
    alpha: float,
    seed: int,
):
    try:
        import matplotlib
        matplotlib.use('Agg')  # non-interactive backend
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print(f'[visualize] Skip visualization (matplotlib not available): {e}')
        return

    # Prepare data
    rng = torch.Generator()
    rng.manual_seed(seed)
    class_ids = sorted(stacked_features.keys())
    all_feats = []
    all_labels = []
    label_offsets = {}
    for cls_id in class_ids:
        feats = stacked_features[cls_id]
        label_offsets[cls_id] = len(all_feats)
        all_feats.append(feats)
        all_labels.append(torch.full((feats.size(0),), cls_id, dtype=torch.long))
    all_feats = torch.cat(all_feats, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # Subsample if too many
    N = all_feats.size(0)
    if N > max_tokens:
        perm = torch.randperm(N, generator=rng)[:max_tokens]
        feats_plot = all_feats[perm]
        labels_plot = all_labels[perm]
    else:
        feats_plot = all_feats
        labels_plot = all_labels

    # Normalize tokens if features were normalized during clustering
    if metadata.get('normalize_features', False):
        feats_plot = F.normalize(feats_plot, dim=1)
    label_unique = sorted(class_ids)

    # Color map: special palette for few classes, otherwise high-contrast
    n_classes = len(label_unique)
    if n_classes <= 5:
        # predefined distinct colors for up to 5 classes
        base_colors = ['red', 'yellow', 'blue', 'black', 'white']
        color_map = {cid: base_colors[i] for i, cid in enumerate(label_unique)}
    else:
        if n_classes <= 20:
            cmap = plt.get_cmap('tab20', n_classes)
        else:
            cmap = plt.get_cmap('hsv', n_classes)
        color_map = {cid: cmap(idx) for idx, cid in enumerate(label_unique)}

    # --- PCA Visualization (Existing) ---
    print('[visualize] Running PCA projection...')
    proj_pca = _pca_project(feats_plot, k=2).cpu()

    # Figure 1: PCA tokens only
    fig, ax = plt.subplots(figsize=(8, 6), dpi=140)
    for cid in label_unique:
        mask = labels_plot == cid
        if mask.any():
            ax.scatter(
                proj_pca[mask, 0],
                proj_pca[mask, 1],
                s=8,
                alpha=alpha,
                label=str(metadata.get('class_names', [''])[cid]) if metadata.get('class_names') else f'class {cid}',
                color=color_map[cid],
            )
    ax.set_title('Token PCA (per-class colors)')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.legend(fontsize=6, ncol=2, frameon=False)
    fig.tight_layout()
    out_path_tokens_pca = os.path.join(out_dir, 'viz_tokens_pca_screen.png')
    fig.savefig(out_path_tokens_pca)
    plt.close(fig)

    # Figure 2: PCA with prototypes
    Xc = feats_plot - feats_plot.mean(0, keepdim=True)
    try:
        _, _, Vh = torch.linalg.svd(Xc, full_matrices=False)
    except RuntimeError:
        Xc_cpu = Xc.cpu()
        _, _, Vh = torch.linalg.svd(Xc_cpu, full_matrices=False)
        Vh = Vh.to(Xc.device)
    comps2 = Vh[:2]

    fig2, ax2 = plt.subplots(figsize=(8, 6), dpi=140)
    ax2.scatter(proj_pca[:, 0], proj_pca[:, 1], s=6, alpha=alpha * 0.4, c=[color_map[int(c.item())] for c in labels_plot])
    for cid in label_unique:
        pinfo = prototypes.get(int(cid))
        if pinfo is None: continue
        centers = pinfo['centers']
        centers_t = torch.as_tensor(centers)
        centers_proj = ((centers_t - feats_plot.mean(0, keepdim=True)) @ comps2.T).cpu()
        ax2.scatter(
            centers_proj[:, 0], centers_proj[:, 1],
            marker='*', s=120, edgecolors='k', linewidths=1.5,  # 增加黑色轮廓
            color=color_map[cid], label=f'proto c{cid}',
        )
        if 'counts' in pinfo:
            for k_idx, (x, y) in enumerate(centers_proj):
                cnts = pinfo['counts']
                if k_idx < len(cnts):
                    ax2.text(x, y, str(int(cnts[k_idx])), fontsize=6, ha='center', va='center')
    ax2.set_title('Token + Prototype Centers (PCA)')
    ax2.set_xlabel('PC1')
    ax2.set_ylabel('PC2')
    fig2.tight_layout()
    out_path_proto_pca = os.path.join(out_dir, 'viz_tokens_with_prototypes_pca_screen.png')
    fig2.savefig(out_path_proto_pca)
    plt.close(fig2)

    # --- t-SNE Visualization (New) ---
    print('[visualize] Running t-SNE projection...')
    
    # Run t-SNE on tokens and prototypes together for a consistent embedding
    proto_centers_list = []
    proto_labels_list = []
    for cid in label_unique:
        pinfo = prototypes.get(int(cid))
        if pinfo is not None:
            centers = pinfo['centers']
            proto_centers_list.append(torch.as_tensor(centers))
            proto_labels_list.extend([cid] * centers.shape[0])

    if proto_centers_list:
        all_proto_centers = torch.cat(proto_centers_list, dim=0).to(feats_plot.device)
        
        # Normalize prototype centers if features were normalized可视化的时候归一化一下
        if metadata.get('normalize_features', False):
            all_proto_centers = F.normalize(all_proto_centers, dim=1)

        all_proto_labels = torch.tensor(proto_labels_list, device=labels_plot.device)

        combined_feats = torch.cat([feats_plot, all_proto_centers], dim=0)
        print(f'[visualize] Running combined t-SNE for {combined_feats.size(0)} points...')
        combined_proj_tsne = _tsne_project(combined_feats, seed=seed).cpu()

        num_plot_tokens = feats_plot.size(0)
        proj_tokens_tsne = combined_proj_tsne[:num_plot_tokens]
        proj_protos_tsne = combined_proj_tsne[num_plot_tokens:]

        # Figure 3: t-SNE tokens only (from combined result)
        fig3, ax3 = plt.subplots(figsize=(8, 6), dpi=140)
        for cid in label_unique:
            mask = labels_plot == cid
            if mask.any():
                ax3.scatter(
                    proj_tokens_tsne[mask, 0], proj_tokens_tsne[mask, 1],
                    s=8, alpha=alpha,
                    label=str(metadata.get('class_names', [''])[cid]) if metadata.get('class_names') else f'class {cid}',
                    color=color_map[cid],
                )
        ax3.set_title('Token t-SNE (per-class colors)')
        ax3.set_xlabel('t-SNE dim 1')
        ax3.set_ylabel('t-SNE dim 2')
        ax3.legend(fontsize=6, ncol=2, frameon=False)
        fig3.tight_layout()
        out_path_tokens_tsne = os.path.join(out_dir, 'viz_tokens_tsne_screen.png')
        fig3.savefig(out_path_tokens_tsne)
        plt.close(fig3)

        # Figure 4: t-SNE with prototypes
        fig4, ax4 = plt.subplots(figsize=(8, 6), dpi=140)
        ax4.scatter(proj_tokens_tsne[:, 0], proj_tokens_tsne[:, 1], s=6, alpha=alpha * 0.4, c=[color_map[int(c.item())] for c in labels_plot])
        ax4.scatter(
            proj_protos_tsne[:, 0], proj_protos_tsne[:, 1],
            marker='*', s=120, edgecolors='k', linewidths=1.5,  # 增加黑色轮廓
            c=[color_map[int(c.item())] for c in all_proto_labels]
        )
        ax4.set_title('Token + Prototype Centers (t-SNE)')
        ax4.set_xlabel('t-SNE dim 1')
        ax4.set_ylabel('t-SNE dim 2')
        fig4.tight_layout()
        out_path_proto_tsne = os.path.join(out_dir, 'viz_tokens_with_prototypes_tsne_screen.png')
        fig4.savefig(out_path_proto_tsne)
        plt.close(fig4)

    # --- Common Visualizations ---
    # Bar chart of counts
    fig_bar, ax_bar = plt.subplots(figsize=(10, 4), dpi=140)
    bars, labels_bar = [], []
    for cid in label_unique:
        pinfo = prototypes.get(int(cid))
        if not pinfo or 'counts' not in pinfo: continue
        counts = pinfo['counts']
        for idx, cval in enumerate(counts):
            bars.append(int(cval))
            labels_bar.append(f'{cid}-{idx}')
    ax_bar.bar(range(len(bars)), bars)
    ax_bar.set_xticks(range(len(bars)))
    ax_bar.set_xticklabels(labels_bar, rotation=90, fontsize=6)
    ax_bar.set_ylabel('Count')
    ax_bar.set_title('Prototype Instance Counts')
    fig_bar.tight_layout()
    out_path_bar = os.path.join(out_dir, 'viz_prototype_counts_screen.png')
    fig_bar.savefig(out_path_bar)
    plt.close(fig_bar)

    # Save projection raw data
    try:
        torch.save({
            'proj_pca': proj_pca,
            'proj_tsne': proj_tokens_tsne if proto_centers_list else None,
            'labels_tokens': labels_plot,
            'class_ids': class_ids,
        }, os.path.join(out_dir, 'viz_projection_tokens_screen.pt'))
    except Exception as e:
        print(f'[visualize] Failed to save projection tensor: {e}')

    try:
        _visualize_prototype_distance_matrix(out_dir, prototypes, metadata)
    except Exception as e:
        print(f'[visualize] Failed to render prototype distance heatmap: {e}')


def _visualize_prototype_distance_matrix(
    out_dir: str,
    prototypes: Dict[int, Dict[str, torch.Tensor]],
    metadata: Dict,
):
    """Create a heatmap for pairwise prototype distances (1 - cosine similarity)."""
    import math
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f'[distance] Skip heatmap (matplotlib import error): {e}')
        return

    # Collect centers
    class_ids = sorted(prototypes.keys())
    center_list, label_list = [], []
    for cid in class_ids:
        pinfo = prototypes[cid]
        centers = pinfo.get('centers_l2') if 'centers_l2' in pinfo else pinfo.get('centers')
        if centers is None: continue
        centers_t = F.normalize(torch.as_tensor(centers).float(), dim=1)
        for idx, row in enumerate(centers_t):
            center_list.append(row)
            label_list.append(f'c{cid}-p{idx}')

    if not center_list:
        print('[distance] No prototype centers collected; skip distance heatmap.')
        return

    centers_all = torch.stack(center_list, dim=0)
    sim = centers_all @ centers_all.t()
    sim = sim.clamp(-1.0, 1.0)
    dist = 1.0 - sim

    # Plot heatmap
    M = dist.size(0)
    fig_size = max(4, min(18, M * 0.28))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size), dpi=120)
    im = ax.imshow(dist.cpu().numpy(), cmap='viridis')
    ax.set_title('Prototype Distance (1 - cosine)')
    ax.set_xticks(range(M))
    ax.set_yticks(range(M))
    if M <= 40:
        ax.set_xticklabels(label_list, rotation=90, fontsize=6)
        ax.set_yticklabels(label_list, fontsize=6)
    else:
        step = max(1, M // 40)
        ax.set_xticks(range(0, M, step))
        ax.set_yticks(range(0, M, step))
        ax.set_xticklabels([label_list[i] for i in range(0, M, step)], rotation=90, fontsize=5)
        ax.set_yticklabels([label_list[i] for i in range(0, M, step)], fontsize=5)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label='1 - cosine')
    fig.tight_layout()
    out_img = os.path.join(out_dir, 'viz_prototype_distance_screen.png')
    fig.savefig(out_img)
    plt.close(fig)
    try:
        torch.save({'distance': dist.cpu(), 'labels': label_list, 'metadata': metadata},
                   os.path.join(out_dir, 'prototype_distance_matrix_screen.pt'))
    except Exception as e:
        print(f'[distance] Failed to save distance matrix tensor: {e}')


def _save_prototypes_report(
    out_dir: str,
    prototypes: Dict[int, Dict[str, torch.Tensor]],
    metadata: Dict,
):
    """Saves a human-readable text file with prototype vectors and similarity."""
    try:
        import numpy as np
    except ImportError:
        print('[report] NumPy not found, cannot generate text report.')
        return

    report_path = os.path.join(out_dir, 'prototypes_report_screen.txt')

    # Collect all normalized centers and labels
    center_list, label_list = [], []
    class_ids = sorted(prototypes.keys())
    class_names = metadata.get('class_names')
    if class_names is None:
        class_names = [f'Class {i}' for i in range(max(class_ids) + 1 if class_ids else 0)]

    for cid in class_ids:
        pinfo = prototypes[cid]
        # Use l2 normalized if available, otherwise compute it
        centers = pinfo.get('centers_l2')
        if centers is None:
            centers = F.normalize(pinfo['centers'], dim=1)

        for i in range(centers.size(0)):
            center_list.append(centers[i])
            label_list.append(f'{class_names[cid]}-{i}')

    if not center_list:
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("No prototypes found to generate a report.\n")
        return

    all_centers = torch.stack(center_list, dim=0)
    similarity_matrix = (all_centers @ all_centers.T).cpu().numpy()

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("Prototype Analysis Report\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Source Config: {metadata.get('config')}\n")
        f.write(f"Source Checkpoint: {metadata.get('checkpoint')}\n")
        f.write(f"Total Instances Extracted: {metadata.get('total_instances')}\n")
        f.write(f"Feature Filtering Enabled: {metadata.get('feature_filtering_enabled', False)}\n")
        if metadata.get('feature_filtering_enabled'):
            f.write(f"Instances After Filtering: {metadata.get('instances_after_filtering', 'N/A')}\n")
        f.write(f"Number of Classes with Prototypes: {len(class_ids)}\n\n")

        f.write("-" * 80 + "\n")
        f.write("Per-Class Prototype Vectors\n")
        f.write("-" * 80 + "\n")
        for cid in class_ids:
            pinfo = prototypes[cid]
            centers = pinfo['centers']
            counts = pinfo.get('counts')
            f.write(f"\nClass: {class_names[cid]} (ID: {cid})\n")
            f.write(f"Instance Count for this class: {metadata['per_class_instance_counts'].get(cid, 'N/A')}\n")
            if metadata.get('filtered_instance_counts'):
                f.write(f"Instance Count after filtering: {metadata['filtered_instance_counts'].get(cid, 'N/A')}\n")
            for i in range(centers.size(0)):
                count_str = f"(support count: {counts[i]})" if counts is not None and i < len(counts) else ""
                vector_str = np.array2string(centers[i].numpy(), precision=4, suppress_small=True, max_line_width=120)
                f.write(f"  - Prototype {i} {count_str}: {vector_str}\n")

        f.write("\n\n" + "=" * 80 + "\n")
        f.write("Inter-Prototype Cosine Similarity Matrix\n")
        f.write("=" * 80 + "\n\n")

        # Format and write matrix
        header = "          " + " ".join([f"{s:<7.7}" for s in label_list])
        f.write(header + "\n")
        for i, row_label in enumerate(label_list):
            row_str = f"{row_label:<10.10}" + " ".join([f"{similarity_matrix[i, j]:>7.3f}" for j in range(similarity_matrix.shape[1])])
            f.write(row_str + "\n")

    print(f"Prototype text report saved to {report_path}")


@torch.no_grad()
def extract_prototypes(args: argparse.Namespace) -> Dict:
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    _maybe_import_custom_modules(cfg)

    cfg_model = cfg.model.copy()
    model = build_detector(cfg_model, test_cfg=cfg.get('test_cfg'))
    device = torch.device(args.device)
    model.to(device)
    model.eval()
    load_checkpoint(model, args.checkpoint, map_location=device)

    dataset_cfg = _ensure_single_dataset(cfg.data.train)
    dataset = build_dataset(dataset_cfg)

    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=args.batch_size,
        workers_per_gpu=args.workers,
        dist=False,
        shuffle=False,
    )

    feature_store: Dict[int, List[torch.Tensor]] = defaultdict(list)
    total_samples = len(dataset)
    if args.max_samples is not None:
        total_samples = min(total_samples, args.max_samples)

    progress = None if args.no_progress else ProgressBar(total_samples)
    processed = 0
    total_instances = 0

    start_time = time.time()

    for data in data_loader:
        if args.max_samples is not None and processed >= args.max_samples:
            break

        imgs = _unwrap_data_container(data['img'])
        if isinstance(imgs, list):
            imgs = torch.stack(imgs, dim=0)
        imgs = imgs.to(device)
        #  推理一两张图
        gt_bboxes_batch = _unwrap_data_container(data['gt_bboxes'])
        gt_labels_batch = _unwrap_data_container(data['gt_labels'])

        batch_size = imgs.size(0)
        for batch_idx in range(batch_size):
            if args.max_samples is not None and processed >= args.max_samples:
                break
            processed += 1
            if progress is not None:
                progress.update()

            gt_bboxes = gt_bboxes_batch[batch_idx]
            gt_labels = gt_labels_batch[batch_idx]

            if gt_bboxes.numel() == 0:
                continue

            batch_img = imgs[batch_idx:batch_idx + 1]
            feats = model.extract_feat(batch_img)

            rois = bbox2roi([gt_bboxes.to(device)])
            rois = rois.to(device)
            bbox_feats = _extract_roi_features(model, feats, rois, device)

            pooled = bbox_feats.mean(dim=[2, 3])
            labels = gt_labels.to(device)

            pooled_cpu = pooled.cpu().float()
            labels_cpu = labels.cpu()
            for feat_vec, cls in zip(pooled_cpu, labels_cpu):
                cls_id = int(cls.item())
                feature_store[cls_id].append(feat_vec)
                total_instances += 1

        if args.max_samples is not None and processed >= args.max_samples:
            break

    elapsed = time.time() - start_time

    stacked_features: Dict[int, torch.Tensor] = {}
    for cls_id, feats in feature_store.items():
        if feats:
            stacked_features[int(cls_id)] = torch.stack(feats).float()

    if not stacked_features:
        raise RuntimeError('No ROI features were collected. Check dataset annotations or configuration.')

    per_class_counts = {cls_id: int(tensor.size(0)) for cls_id, tensor in stacked_features.items()}
    feature_dim = next(iter(stacked_features.values())).size(1)

    # 如果启用了过滤，则在此处应用
    filtered_instance_counts = None
    instances_after_filtering = total_instances
    if args.enable_filtering:
        print("\n" + "="*50)
        print("Applying feature filtering...")
        print("="*50)
        # 调用过滤函数，处理所有类别的特征
        stacked_features = apply_feature_filtering(
            stacked_features,
            filter_outliers_ratio=args.filter_outliers_ratio,
            filter_knn_ratio=args.filter_knn_ratio,
            filter_activation_ratio=args.filter_activation_ratio,
            verbose=not args.no_progress
        )
        
        # 更新过滤后的统计信息
        filtered_instance_counts = {cls_id: int(tensor.size(0)) for cls_id, tensor in stacked_features.items()}
        instances_after_filtering = sum(filtered_instance_counts.values())
        print(f"\nSummary: {total_instances} -> {instances_after_filtering} instances "
              f"(kept {instances_after_filtering/total_instances*100:.1f}%)")
        print("="*50 + "\n")

    if args.save_tokens:
        token_payload = {
            'features': {cls_id: tensor for cls_id, tensor in stacked_features.items()},
            'class_names': getattr(dataset, 'CLASSES', None),
            'metadata': {
                'config': args.config,
                'checkpoint': args.checkpoint,
                'feature_dim': feature_dim,
                'normalize_features': args.normalize_features,
                'feature_filtering_enabled': args.enable_filtering,
                'instances_after_filtering': instances_after_filtering if args.enable_filtering else None,
            }
        }
        save_dir = os.path.dirname(args.save_tokens)
        if save_dir:
            mmcv.mkdir_or_exist(save_dir)
        torch.save(token_payload, args.save_tokens)

    prototypes_dict, cluster_metrics = compute_prototypes_sinkhorn(
        stacked_features,
        num_prototypes=args.num_prototypes,
        normalize_features=args.normalize_features,
        device=device,
        epochs=args.sinkhorn_epochs,
        batch_size=args.sinkhorn_batch_size,
        queue_size=args.sinkhorn_queue_size,
        momentum=args.sinkhorn_momentum,
        iterations=args.sinkhorn_iterations,
        epsilon=args.sinkhorn_epsilon,
    )

    for entry in prototypes_dict.values():
        entry['method'] = 'sinkhorn'
        if args.l2_normalize:
            entry['centers_l2'] = F.normalize(entry['centers'], dim=1)

    metadata = {
        'config': args.config,
        'checkpoint': args.checkpoint,
        'device': str(device),
        'samples_processed': processed,
        'elapsed_seconds': elapsed,
        'class_names': getattr(dataset, 'CLASSES', None),
        'feature_dim': int(feature_dim),
        'cluster_method': 'sinkhorn',
        'num_prototypes': args.num_prototypes,
        'normalize_features': args.normalize_features,
        'l2_normalized_copy': args.l2_normalize,
        'total_instances': total_instances,
        'per_class_instance_counts': per_class_counts,
        'cluster_metrics': cluster_metrics,
        'tokens_saved_to': args.save_tokens,
        'feature_filtering_enabled': args.enable_filtering,
        'instances_after_filtering': instances_after_filtering if args.enable_filtering else None,
        'filtered_instance_counts': filtered_instance_counts if args.enable_filtering else None,
    }

    output = {
        'metadata': metadata,
        'prototypes': prototypes_dict,
    }

    output_path = args.out
    if output_path is None:
        ckpt_dir = os.path.dirname(args.checkpoint)
        output_path = os.path.join(ckpt_dir, 'class_prototypes_screen.pth')
    output_dir = os.path.dirname(output_path)
    if output_dir:
        mmcv.mkdir_or_exist(output_dir)
    else:
        mmcv.mkdir_or_exist('.')
    torch.save(output, output_path)
    
    summary_msg = (
        f'Prototype statistics saved to {output_path}\n'
        f'Processed {processed} images, collected {total_instances} instances'
    )
    if args.enable_filtering:
        summary_msg += f', filtered to {instances_after_filtering} instances'
    summary_msg += f' in {elapsed:.2f}s'
    print(summary_msg)

    # Generate text report
    try:
        _save_prototypes_report(output_dir if output_dir else '.', prototypes_dict, metadata)
    except Exception as e:
        print(f'[report] Failed to generate prototype text report: {e}')

    # Visualization (unless disabled)
    if not args.no_visualize:
        viz_dir = output_dir if output_dir else '.'
        try:
            _visualize_tokens_and_prototypes(
                viz_dir,
                stacked_features,
                prototypes_dict,
                metadata,
                max_tokens=args.visual_max_tokens,
                alpha=args.visual_alpha,
                seed=args.visual_seed,
            )
            print(f'Visualization figures saved under: {viz_dir}')
        except Exception as e:  # pragma: no cover
            print(f'[visualize] Failed to render prototype visualizations: {e}')

    return output


def main() -> None:
    args = parse_args()
    extract_prototypes(args)


if __name__ == '__main__':
    main()

