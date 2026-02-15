#!/usr/bin/env python3
"""基于骨干网络特征的频域原型提取脚本。

流程：加载已训练的检测器，截取骨干某一 stage（默认 C5）的特征图，
对每个特征图执行 2D FFT -> 频谱幅值 -> 频谱中心化/对数压缩 -> 自适应池化，
并根据通道聚合策略生成固定维度向量，随后通过 Sinkhorn 聚类得到频域原型。

该脚本与 `extract_source_prototypes_frequency.py` 并列，可分别对比图像频域与特征频域原型。"""
import argparse
import os
import sys
import time
from copy import deepcopy
from typing import Dict, List, Tuple, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, '.')

import mmcv
import torch
import torch.nn.functional as F
from mmcv import Config, DictAction
from mmcv.runner import load_checkpoint
from mmcv.utils import ProgressBar

from mmdet.datasets import build_dataset, build_dataloader
from mmdet.models import build_detector

from prototype_tools.extract_source_prototypes_global import (  # type: ignore
    compute_prototypes_sinkhorn,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='从骨干网络特征生成频域原型（默认使用 Swin C5 stage）。'
    )
    parser.add_argument('--config', required=True, help='mmdet 训练配置文件。')
    parser.add_argument('--checkpoint', required=True, help='已训练模型权重 (.pth)。')
    parser.add_argument('--out', required=True, help='输出原型文件路径 (.pth)。')
    parser.add_argument('--device', default='cuda:0', help='特征提取设备。')
    parser.add_argument('--batch-size', type=int, default=8, help='特征提取批量。')
    parser.add_argument('--workers', type=int, default=4, help='DataLoader worker 数。')
    parser.add_argument('--max-samples', type=int, default=None, help='调试用，最多处理的图像数。')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction, help='覆盖配置项。')
    parser.add_argument('--no-progress', action='store_true', help='关闭进度条。')

    parser.add_argument('--feature-stage', default='c5', choices=['c2', 'c3', 'c4', 'c5', 'last'],
                        help='选择骨干输出的哪一层作为频域特征输入。')
    parser.add_argument('--freq-pool-size', type=int, default=24,
                        help='自适应池化的空间尺寸，输出向量长度与通道聚合方式相关。')
    parser.add_argument('--channel-aggregation', choices=['mean', 'max', 'stack'], default='mean',
                        help='通道聚合方式，stack 会保留全部通道（可能导致维度极大）。')
    parser.add_argument('--channel-subsample', type=int, default=None,
                        help='可选，对通道维度做子采样（仅 stack 时生效，取前 N 个通道）。')
    parser.add_argument('--fft-log-amplitude', action='store_true',
                        help='对幅值谱做 log1p 压缩。默认开启。')
    parser.add_argument('--fft-no-log', dest='fft_log_amplitude', action='store_false')
    parser.set_defaults(fft_log_amplitude=True)
    parser.add_argument('--fft-shift', action='store_true', help='是否执行 fftshift。默认开启。')
    parser.add_argument('--fft-no-shift', dest='fft_shift', action='store_false')
    parser.set_defaults(fft_shift=True)
    parser.add_argument('--normalize-features', dest='normalize_features', action='store_true',
                        help='聚类前做 L2 归一化。')
    parser.add_argument('--no-normalize-features', dest='normalize_features', action='store_false')
    parser.set_defaults(normalize_features=True)

    parser.add_argument('--num-prototypes', type=int, default=16, help='聚类原型数量。')
    parser.add_argument('--sinkhorn-epochs', type=int, default=120)
    parser.add_argument('--sinkhorn-batch-size', type=int, default=512)
    parser.add_argument('--sinkhorn-queue-size', type=int, default=8192)
    parser.add_argument('--sinkhorn-momentum', type=float, default=0.02)
    parser.add_argument('--sinkhorn-iterations', type=int, default=5)
    parser.add_argument('--sinkhorn-epsilon', type=float, default=1e-2)

    parser.add_argument('--save-tokens', default=None,
                        help='可选，保存所有频域特征向量 (.pth)。')
    return parser.parse_args()


def _maybe_import_custom_modules(cfg: Config) -> None:
    custom_imports_cfg = cfg.get('custom_imports')
    if not custom_imports_cfg:
        return
    from mmcv.utils import import_modules_from_strings
    import_modules_from_strings(**custom_imports_cfg)


def _ensure_single_dataset(train_cfg: Dict) -> Dict:
    cfg = deepcopy(train_cfg)
    if isinstance(cfg, dict) and cfg.get('type') == 'RepeatDataset':
        cfg = cfg['dataset']
    return cfg


def _unwrap_data_container(container):
    from mmcv.parallel import DataContainer

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


def _select_stage(backbone_out, stage: str) -> torch.Tensor:
    if isinstance(backbone_out, (list, tuple)):
        mapping = {'c2': 0, 'c3': 1, 'c4': 2, 'c5': 3}
        if stage == 'last':
            idx = len(backbone_out) - 1
        else:
            idx = mapping.get(stage, len(backbone_out) - 1)
            idx = min(idx, len(backbone_out) - 1)
        feat = backbone_out[idx]
    elif isinstance(backbone_out, dict):
        keys = list(backbone_out.keys())
        if stage == 'last':
            key = keys[-1]
        else:
            mapping = {'c2': 0, 'c3': 1, 'c4': 2, 'c5': 3}
            key = keys[mapping.get(stage, len(keys) - 1)]
        feat = backbone_out[key]
    else:
        feat = backbone_out
    if not torch.is_tensor(feat):
        raise TypeError(f'Backbone stage output应为Tensor，当前类型: {type(feat)}')
    return feat


def _freq_vector_from_feature(
    feature: torch.Tensor,
    pool_size: int,
    log_amplitude: bool,
    do_shift: bool,
    channel_agg: str,
    channel_subsample: Optional[int],
) -> torch.Tensor:
    if feature.dim() != 4 or feature.size(0) != 1:
        raise ValueError('feature 必须是形状 (1, C, H, W) 的张量。')
    feat = feature
    if channel_agg == 'stack' and channel_subsample is not None:
        feat = feat[:, :channel_subsample]

    freq_map = torch.fft.fft2(feat, dim=(-2, -1))
    magnitude = torch.abs(freq_map)
    if do_shift:
        magnitude = torch.fft.fftshift(magnitude, dim=(-2, -1))
    if log_amplitude:
        magnitude = torch.log1p(magnitude)

    if channel_agg == 'mean':
        magnitude = magnitude.mean(dim=1, keepdim=True)
    elif channel_agg == 'max':
        magnitude = magnitude.amax(dim=1, keepdim=True)
    elif channel_agg == 'stack':
        pass
    else:
        raise ValueError(f'Unsupported channel aggregation: {channel_agg}')

    if pool_size > 0:
        magnitude = F.adaptive_avg_pool2d(magnitude, (pool_size, pool_size))

    vector = magnitude.reshape(magnitude.size(0), -1)
    if channel_agg != 'stack':
        vector = vector.squeeze(0)
    return vector.reshape(-1)


@torch.no_grad()
def extract_backbone_frequency_prototypes(args: argparse.Namespace) -> Dict:
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    _maybe_import_custom_modules(cfg)

    model = build_detector(cfg.model.copy(), test_cfg=cfg.get('test_cfg'))
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
        backbone_feats = model.backbone(imgs)
        stage_feat = _select_stage(backbone_feats, args.feature_stage)

        if isinstance(stage_feat, (list, tuple)):
            # 一些骨干可能返回多尺度列表，再取最后一项
            stage_feat = stage_feat[-1]
        stage_feat = stage_feat.to(device)

        for b in range(stage_feat.size(0)):
            if args.max_samples is not None and processed >= args.max_samples:
                break
            processed += 1
            if progress is not None:
                progress.update()

            single_feat = stage_feat[b:b + 1]
            freq_vec = _freq_vector_from_feature(
                single_feat,
                pool_size=args.freq_pool_size,
                log_amplitude=args.fft_log_amplitude,
                do_shift=args.fft_shift,
                channel_agg=args.channel_aggregation,
                channel_subsample=args.channel_subsample,
            )
            feature_list.append(freq_vec.cpu().float())

    elapsed = time.time() - start_time

    if not feature_list:
        raise RuntimeError('未收集到任何频域特征，请检查配置。')

    features_tensor = torch.stack(feature_list, dim=0)
    total_images = features_tensor.size(0)
    feature_dim = features_tensor.size(1)

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

    freq_proto = prototypes_dict[0]
    freq_proto['method'] = 'sinkhorn'

    mean_vec = features_tensor.mean(dim=0)
    var_vec = features_tensor.var(dim=0, unbiased=False)

    metadata = {
        'config': args.config,
        'checkpoint': args.checkpoint,
        'device': str(device),
        'samples_processed': processed,
        'elapsed_seconds': elapsed,
        'feature_stage': args.feature_stage,
        'freq_pool_size': args.freq_pool_size,
        'channel_aggregation': args.channel_aggregation,
        'channel_subsample': args.channel_subsample,
        'fft_log_amplitude': args.fft_log_amplitude,
        'fft_shift': args.fft_shift,
        'normalize_features': args.normalize_features,
        'feature_dim': int(feature_dim),
        'num_prototypes': args.num_prototypes,
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

    metadata['input_source'] = 'backbone'

    output = {
        'metadata': metadata,
        'freq_proto': freq_proto,
        'frequency_stats': {
            'mean': mean_vec,
            'var': var_vec,
        },
    }

    out_dir = os.path.dirname(args.out)
    if out_dir:
        mmcv.mkdir_or_exist(out_dir)
    torch.save(output, args.out)

    print(
        f"\nBackbone frequency prototype statistics saved to {args.out}\n"
        f"Processed {processed} items in {elapsed:.2f}s (feature_dim={feature_dim})"
    )

    if args.save_tokens:
        token_payload = {
            'features': features_tensor,
            'metadata': {k: v for k, v in metadata.items() if k not in ['cluster_metrics', 'sinkhorn']},
        }
        torch.save(token_payload, args.save_tokens)
        print(f'Frequency feature tokens saved to {args.save_tokens}')

    return output


def main() -> None:
    args = parse_args()
    extract_backbone_frequency_prototypes(args)


if __name__ == '__main__':
    main()
