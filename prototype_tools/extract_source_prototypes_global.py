#!/usr/bin/env python3
"""提取跨域适配所需的图级原型。
图级原型提取
流程：加载 mmdetection 检测器与权重，遍历源域训练集，对每张图像提取全局特征向量，
随后使用 Sinkhorn 原型学习算法对所有图级特征聚类，生成若干图级原型并写入磁盘。

特征来源可选择 FPN 的最高层 (P5) 或骨干的最后一个 stage (C4)，
默认对 pooled 特征做 L2 归一化后聚类。
"""
import argparse
import os
import sys
import time
from copy import deepcopy
from typing import Dict, List, Tuple

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

from mmdet.datasets import build_dataset, build_dataloader
from mmdet.models import build_detector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='从已训练的源域检测器中提取图级全局特征并聚类为原型。'
    )
    parser.add_argument(
        '--config',
        default='./work_dirs/Omni_Tuner_configs/swin-l_xview3c/xview3c_retinanet_swin_large_1x_myvpt2.py',
        help='mmdet 训练用的配置文件路径。')
    parser.add_argument(
        '--checkpoint',
        default='./work_dirs/work_dirs/xview3c_retinanet_swin_large_1x_myvpt2/best.pth',
        help='已训练模型权重 (.pth) 文件路径。')
    parser.add_argument(
        '--out',
        default='./work_dirs/work_dirs/xview3c_retinanet_swin_large_1x_myvpt2/xview3c_prototypes_global.pth',
        help='图级原型统计输出路径 (.pth)。')
    parser.add_argument('--device', default='cuda:0', help='用于特征提取的设备（如 cuda:0 或 cpu）。')
    parser.add_argument('--batch-size', type=int, default=16, help='特征提取时每 GPU 的样本数。')
    parser.add_argument('--workers', type=int, default=8, help='数据加载的工作线程数。')
    parser.add_argument('--max-samples', type=int, default=None, help='最多处理的图片数量（用于快速测试）。')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction, help='覆盖配置文件中的设置。')
    parser.add_argument('--no-progress', action='store_true', help='关闭进度条显示。')

    # 全局特征相关参数
    parser.add_argument('--feature-source', choices=['p5', 'c4'], default='c4',
                        help='选择从 FPN 最高层 (p5) 或骨干最后 stage (c4) 提取图级特征。')
    parser.add_argument('--pool-type', choices=['avg', 'avgmax'], default='avg',
                        help='图级特征池化方式：avg=全局平均；avgmax=平均拼接最大。')
    parser.add_argument('--normalize-features', dest='normalize_features', action='store_true',
                        help='聚类前对特征做 L2 归一化（默认开启）。')
    parser.add_argument('--no-normalize-features', dest='normalize_features', action='store_false',
                        help='聚类前不做特征归一化。')
    parser.set_defaults(normalize_features=True)
    parser.add_argument('--l2-normalize', action='store_true', help='为每个原型额外保存L2归一化副本。')

    # Sinkhorn 聚类参数
    parser.add_argument('--num-prototypes', type=int, default=12, help='输出的图级原型数量。')
    parser.add_argument('--sinkhorn-epochs', type=int, default=100, help='Sinkhorn 原型学习的训练轮数。')
    parser.add_argument('--sinkhorn-batch-size', type=int, default=512, help='Sinkhorn 原型学习的批量大小。')
    parser.add_argument('--sinkhorn-queue-size', type=int, default=8192, help='Sinkhorn 原型学习的队列长度。')
    parser.add_argument('--sinkhorn-momentum', type=float, default=0.02, help='Sinkhorn 原型更新的动量。')
    parser.add_argument('--sinkhorn-iterations', type=int, default=5, help='每步 Sinkhorn 迭代次数。')
    parser.add_argument('--sinkhorn-epsilon', type=float, default=1e-2, help='Sinkhorn 迭代的熵正则权重。')

    parser.add_argument('--save-tokens', default=None,
                        help='可选，保存所有图级特征向量的路径 (.pth)，便于后处理。')
    # 可视化参数
    parser.add_argument('--no-visualize', action='store_true', help='关闭可视化生成（PCA/t-SNE/报告）。')
    parser.add_argument('--visual-max-tokens', type=int, default=4000, help='可视化时最多采样的图像特征数量。')
    parser.add_argument('--visual-alpha', type=float, default=0.55, help='散点图的透明度。')
    parser.add_argument('--visual-seed', type=int, default=0, help='可视化采样的随机种子。')
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
    cfg = deepcopy(train_cfg)
    if isinstance(cfg, dict) and cfg.get('type') == 'RepeatDataset':
        cfg = cfg['dataset']
    return cfg


_SINGLE_STAGE_BACKBONE_CACHE: Dict[str, torch.nn.Module] = {}


def _global_pool(feature_map: torch.Tensor, pool_type: str = 'avg') -> torch.Tensor:
    if feature_map.dim() != 4:
        raise ValueError(f'Expected 4D feature map, got shape {tuple(feature_map.shape)}')
    if pool_type == 'avg':
        pooled = feature_map.mean(dim=(2, 3))
    elif pool_type == 'avgmax':
        pooled_avg = feature_map.mean(dim=(2, 3))
        pooled_max = feature_map.amax(dim=(2, 3))
        pooled = torch.cat([pooled_avg, pooled_max], dim=1)
    else:
        raise ValueError(f'Unsupported pool type: {pool_type}')
    return pooled


def _select_feature_map(model, imgs: torch.Tensor, source: str) -> torch.Tensor:
    if source == 'p5':
        feats = model.extract_feat(imgs)
        if isinstance(feats, torch.Tensor):
            feature_map = feats
        elif isinstance(feats, (list, tuple)):
            feature_map = feats[-1]
        elif isinstance(feats, dict):
            last_key = list(feats.keys())[-1]
            feature_map = feats[last_key]
        else:
            raise TypeError(f'Unexpected feature type from extract_feat: {type(feats)}')
    else:  # c4
        backbone_feats = model.backbone(imgs)
        if isinstance(backbone_feats, torch.Tensor):
            feature_map = backbone_feats
        elif isinstance(backbone_feats, (list, tuple)):
            feature_map = backbone_feats[-1]
        elif isinstance(backbone_feats, dict):
            last_key = list(backbone_feats.keys())[-1]
            feature_map = backbone_feats[last_key]
        else:
            raise TypeError(f'Unexpected feature type from backbone: {type(backbone_feats)}')
    if not isinstance(feature_map, torch.Tensor):
        raise TypeError(f'Selected feature is not a tensor: {type(feature_map)}')
    return feature_map


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
# VISUALIZATION FUNCTIONS (NEWLY ADDED AND MODIFIED)
# ======================================================================

def _pca_project(feature_tensor: torch.Tensor, k: int = 2) -> torch.Tensor:
    if feature_tensor.size(1) <= k:
        return feature_tensor[:, :k]
    X = feature_tensor - feature_tensor.mean(0, keepdim=True)
    try:
        U, S, Vh = torch.linalg.svd(X, full_matrices=False)
    except RuntimeError:
        X_cpu = X.cpu()
        U, S, Vh = torch.linalg.svd(X_cpu, full_matrices=False)
        U = U.to(feature_tensor.device)
        Vh = Vh.to(feature_tensor.device)
    comps = Vh[:k]
    return X @ comps.T


def _tsne_project(feature_tensor: torch.Tensor, k: int = 2, seed: int = 0) -> torch.Tensor:
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        print('[visualize] scikit-learn not available, skipping t-SNE.')
        return torch.zeros(feature_tensor.size(0), k)

    X = feature_tensor.cpu().numpy()
    n_samples = X.shape[0]
    perplexity = min(30.0, float(n_samples - 1))
    if perplexity <= 1.0:
        print(f'[visualize] Not enough samples ({n_samples}) for t-SNE, skipping.')
        return torch.zeros(feature_tensor.size(0), k)

    tsne = TSNE(n_components=k, perplexity=perplexity, n_iter=300,
                random_state=seed, init='pca', learning_rate='auto')
    proj = tsne.fit_transform(X)
    return torch.from_numpy(proj)


def _visualize_prototype_distance_matrix(
    out_dir: str,
    prototypes: Dict[int, Dict[str, torch.Tensor]],
):
    """Create a heatmap for pairwise prototype distances (1 - cosine similarity)."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f'[distance] Skip heatmap (matplotlib import error): {e}')
        return

    # Collect centers for the single "global" class (ID 0)
    pinfo = prototypes.get(0, {})
    centers = pinfo.get('centers_l2') if 'centers_l2' in pinfo else pinfo.get('centers')
    if centers is None:
        print('[distance] No prototype centers found; skip distance heatmap.')
        return

    centers_t = F.normalize(torch.as_tensor(centers).float(), dim=1)
    label_list = [f'p{idx}' for idx in range(centers_t.size(0))]

    sim = centers_t @ centers_t.t()
    sim = sim.clamp(-1.0, 1.0)
    dist = 1.0 - sim

    # Plot heatmap
    M = dist.size(0)
    fig_size = max(4, min(18, M * 0.4))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size), dpi=120)
    im = ax.imshow(dist.cpu().numpy(), cmap='viridis')
    ax.set_title('Global Prototype Distance (1 - Cosine Similarity)')
    ax.set_xticks(range(M))
    ax.set_yticks(range(M))
    ax.set_xticklabels(label_list, rotation=90, fontsize=8)
    ax.set_yticklabels(label_list, fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out_img = os.path.join(out_dir, 'viz_global_prototype_distance.png')
    fig.savefig(out_img)
    plt.close(fig)
    print(f'[visualize] Prototype distance heatmap saved to {out_img}')

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

    report_path = os.path.join(out_dir, 'global_prototypes_report.txt')
    pinfo = prototypes.get(0, {})
    centers_orig = pinfo.get('centers')
    counts = pinfo.get('counts')

    if centers_orig is None:
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("No prototypes found to generate a report.\n")
        return

    centers_norm = F.normalize(centers_orig, dim=1)
    label_list = [f'Prototype-{i}' for i in range(centers_norm.size(0))]
    similarity_matrix = (centers_norm @ centers_norm.T).cpu().numpy()

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("Global Prototype Analysis Report\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Source Config: {metadata.get('config')}\n")
        f.write(f"Source Checkpoint: {metadata.get('checkpoint')}\n")
        f.write(f"Total Images Processed: {metadata.get('samples_processed')}\n\n")

        f.write("-" * 80 + "\n")
        f.write("Global Prototype Vectors\n")
        f.write("-" * 80 + "\n")
        for i in range(centers_orig.size(0)):
            count_str = f"(support count: {counts[i]})" if counts is not None and i < len(counts) else ""
            vector_str = np.array2string(centers_orig[i].numpy(), precision=4, suppress_small=True, max_line_width=120)
            f.write(f" - {label_list[i]} {count_str}:\n   {vector_str}\n\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("Inter-Prototype Cosine Similarity Matrix\n")
        f.write("=" * 80 + "\n\n")

        header = " " * 12 + " ".join([f"{s:<11.11}" for s in label_list])
        f.write(header + "\n")
        for i, row_label in enumerate(label_list):
            row_str = f"{row_label:<12.12}" + " ".join([f"{similarity_matrix[i, j]:>11.4f}" for j in range(similarity_matrix.shape[1])])
            f.write(row_str + "\n")

    print(f"Prototype text report saved to {report_path}")

def _visualize_global_features_and_prototypes(
    out_dir: str,
    features: torch.Tensor,
    prototypes_dict: Dict,
    metadata: Dict,
    max_tokens: int,
    alpha: float,
    seed: int,
):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f'[visualize] Skip visualization (matplotlib not available): {e}')
        return

    rng = torch.Generator()
    rng.manual_seed(seed)

    # Subsample if too many features for plotting
    if features.size(0) > max_tokens:
        perm = torch.randperm(features.size(0), generator=rng)[:max_tokens]
        feats_plot = features[perm]
    else:
        feats_plot = features

    if metadata.get('normalize_features', False):
        feats_plot = F.normalize(feats_plot, dim=1)

    prototypes = prototypes_dict.get(0, {}) # Get prototypes for the single global class
    proto_centers = prototypes.get('centers')
    if proto_centers is not None:
        if metadata.get('normalize_features', False):
            proto_centers = F.normalize(proto_centers, dim=1)

    # --- PCA Visualization ---
    print('[visualize] Running PCA projection...')
    # Project tokens and prototypes together for consistent alignment
    combined_feats_pca = torch.cat([feats_plot, proto_centers], dim=0) if proto_centers is not None else feats_plot
    proj_combined_pca = _pca_project(combined_feats_pca, k=2).cpu()
    proj_tokens_pca = proj_combined_pca[:feats_plot.size(0)]
    proj_protos_pca = proj_combined_pca[feats_plot.size(0):] if proto_centers is not None else None

    fig, ax = plt.subplots(figsize=(8, 6), dpi=140)
    ax.scatter(proj_tokens_pca[:, 0], proj_tokens_pca[:, 1], s=8, alpha=alpha, color='tab:blue', label='Image Features')
    if proj_protos_pca is not None:
        ax.scatter(proj_protos_pca[:, 0], proj_protos_pca[:, 1], marker='*', s=160,
                    edgecolors='k', linewidths=1.4, color='tab:orange', label='Prototypes')
    ax.set_title('Global Features and Prototypes (PCA)')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.legend(fontsize=8, frameon=True)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'viz_global_features_pca.png'))
    plt.close(fig)

    # --- t-SNE Visualization ---
    print('[visualize] Running t-SNE projection...')
    combined_feats_tsne = torch.cat([feats_plot, proto_centers], dim=0) if proto_centers is not None else feats_plot
    proj_combined_tsne = _tsne_project(combined_feats_tsne, seed=seed).cpu()
    proj_tokens_tsne = proj_combined_tsne[:feats_plot.size(0)]
    proj_protos_tsne = proj_combined_tsne[feats_plot.size(0):] if proto_centers is not None else None

    fig, ax = plt.subplots(figsize=(8, 6), dpi=140)
    ax.scatter(proj_tokens_tsne[:, 0], proj_tokens_tsne[:, 1], s=8, alpha=alpha, color='tab:blue', label='Image Features')
    if proj_protos_tsne is not None:
         ax.scatter(proj_protos_tsne[:, 0], proj_protos_tsne[:, 1], marker='*', s=160,
                    edgecolors='k', linewidths=1.4, color='tab:orange', label='Prototypes')
    ax.set_title('Global Features and Prototypes (t-SNE)')
    ax.set_xlabel('t-SNE dim 1')
    ax.set_ylabel('t-SNE dim 2')
    ax.legend(fontsize=8, frameon=True)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'viz_global_features_tsne.png'))
    plt.close(fig)

    print(f'[visualize] PCA and t-SNE figures saved to {out_dir}')

    # --- Bar chart of counts ---
    counts = prototypes.get('counts')
    if counts is not None:
        fig_bar, ax_bar = plt.subplots(figsize=(max(6, counts.size(0) * 0.5), 4), dpi=140)
        labels_bar = [f'P{i}' for i in range(len(counts))]
        bars = counts.cpu().numpy()
        ax_bar.bar(range(len(bars)), bars)
        ax_bar.set_xticks(range(len(bars)))
        ax_bar.set_xticklabels(labels_bar, rotation=45, fontsize=8)
        ax_bar.set_ylabel('Image Count')
        ax_bar.set_title('Global Prototype Support Counts')
        fig_bar.tight_layout()
        out_path_bar = os.path.join(out_dir, 'viz_global_prototype_counts.png')
        fig_bar.savefig(out_path_bar)
        plt.close(fig_bar)
        print(f'[visualize] Prototype counts bar chart saved to {out_path_bar}')

@torch.no_grad()
def extract_global_prototypes(args: argparse.Namespace) -> Dict:
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

    feature_list: List[torch.Tensor] = []
    total_samples = len(dataset)
    if args.max_samples is not None:
        total_samples = min(total_samples, args.max_samples)

    progress = None if args.no_progress else ProgressBar(total_samples)
    processed = 0

    start_time = time.time()

    for data in data_loader:
        if args.max_samples is not None and processed >= args.max_samples:
            break

        imgs = _unwrap_data_container(data['img'])
        if isinstance(imgs, list):
            imgs = torch.stack(imgs, dim=0)
        imgs = imgs.to(device)

        # Process one image at a time to ensure correct feature extraction
        for batch_idx in range(imgs.size(0)):
            if args.max_samples is not None and processed >= args.max_samples:
                break
            processed += 1
            if progress is not None:
                progress.update()

            single_img = imgs[batch_idx:batch_idx + 1]
            feature_map = _select_feature_map(model, single_img, args.feature_source)
            pooled = _global_pool(feature_map, args.pool_type)
            feature_list.append(pooled.cpu().float())

    elapsed = time.time() - start_time

    if not feature_list:
        raise RuntimeError('No global features were collected. Check dataset or configuration.')

    features_tensor = torch.cat(feature_list, dim=0)
    total_images = features_tensor.size(0)
    feature_dim = features_tensor.size(1)

    # For Sinkhorn clustering, we treat all global features as a single class (ID 0)
    features_for_cluster = F.normalize(features_tensor, dim=1) if args.normalize_features else features_tensor
    features_per_class = {0: features_for_cluster}
    prototypes_dict, cluster_metrics = compute_prototypes_sinkhorn(
        features_per_class,
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

    global_proto = prototypes_dict[0]
    global_proto['method'] = 'sinkhorn'
    if args.l2_normalize:
        global_proto['centers_l2'] = F.normalize(global_proto['centers'], dim=1)


    mean_vec = features_tensor.mean(dim=0, keepdim=False)
    var_vec = features_tensor.var(dim=0, unbiased=False)

    metadata = {
        'config': args.config,
        'checkpoint': args.checkpoint,
        'device': str(device),
        'samples_processed': processed,
        'elapsed_seconds': elapsed,
        'feature_source': args.feature_source,
        'pool_type': args.pool_type,
        'normalize_features': args.normalize_features,
        'l2_normalized_copy': args.l2_normalize,
        'feature_dim': int(feature_dim),
        'num_prototypes': args.num_prototypes,
        'cluster_method': 'sinkhorn',
        'cluster_metrics': cluster_metrics,
        'total_images': int(total_images),
        'sinkhorn': {
            'epochs': args.sinkhorn_epochs,
            'batch_size': args.sinkhorn_batch_size,
            'queue_size': args.sinkhorn_queue_size,
            'momentum': args.sinkhorn_momentum,
            'iterations': args.sinkhorn_iterations,
            'epsilon': args.sinkhorn_epsilon,
        },
        'tokens_saved_to': args.save_tokens,
    }

    output = {
        'metadata': metadata,
        'global_proto': global_proto,
        'global_stats': {
            'mean': mean_vec,
            'var': var_vec,
        },
    }

    output_path = args.out
    output_dir = os.path.dirname(output_path)
    if output_dir:
        mmcv.mkdir_or_exist(output_dir)
    torch.save(output, output_path)

    print(
        f'\nGlobal prototype statistics saved to {output_path}\n'
        f'Processed {processed} images in {elapsed:.2f}s (feature_dim={feature_dim})'
    )

    if args.save_tokens:
        token_payload = {
            'features': features_tensor,
            'metadata': {k: v for k, v in metadata.items() if k not in ['cluster_metrics', 'sinkhorn']}
        }
        torch.save(token_payload, args.save_tokens)
        print(f'Global feature tokens saved to {args.save_tokens}')

    # --- Call visualization and reporting functions ---
    if not args.no_visualize:
        viz_dir = output_dir if output_dir else '.'
        print("\n--- Generating Visualizations and Reports ---")
        try:
            # Pass the dictionary of prototypes {0: proto_info}
            _visualize_global_features_and_prototypes(
                viz_dir,
                features_tensor,
                prototypes_dict,
                metadata,
                max_tokens=args.visual_max_tokens,
                alpha=args.visual_alpha,
                seed=args.visual_seed,
            )
            _visualize_prototype_distance_matrix(viz_dir, prototypes_dict)
            _save_prototypes_report(viz_dir, prototypes_dict, metadata)

        except Exception as e:
            print(f'[visualize] Failed to render visualizations: {e}')

    return output


def main() -> None:
    args = parse_args()
    extract_global_prototypes(args)


if __name__ == '__main__':
    main()