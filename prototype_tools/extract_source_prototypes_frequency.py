#!/usr/bin/env python3
"""提取遥感图像在频域下的图级原型。

流程：使用已训练检测器的数据管道加载源域图像，对每张图做 2D FFT，
将幅度谱经过可选的 log/shift 和自适应池化后展平为向量，再使用
Sinkhorn 聚类学习频域原型并写入磁盘。
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

from mmdet.datasets import build_dataset, build_dataloader
from mmdet.models import build_detector

# 复用 Sinkhorn 聚类实现
from prototype_tools.extract_source_prototypes_global import (  # type: ignore
    compute_prototypes_sinkhorn,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='从遥感图像中提取频域全局特征并聚类为原型。'
    )
    parser.add_argument('--config', required=True, help='mmdet 训练配置文件路径。')
    parser.add_argument('--checkpoint', required=True, help='已训练模型权重 (.pth) 路径。')
    parser.add_argument('--out', required=True, help='频域原型输出路径 (.pth)。')
    parser.add_argument('--device', default='cuda:0', help='用于特征提取的设备。')
    parser.add_argument('--batch-size', type=int, default=8, help='特征提取时每 GPU 的样本数。')
    parser.add_argument('--workers', type=int, default=4, help='DataLoader 工作线程数。')
    parser.add_argument('--max-samples', type=int, default=None, help='最多处理的图片数量（调试用）。')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction, help='覆盖配置文件中的设置。')
    parser.add_argument('--no-progress', action='store_true', help='关闭进度条显示。')

    # 频域特征相关
    parser.add_argument('--freq-pool-size', type=int, default=48,
                        help='自适应池化的目标尺寸，输出向量长度为 (C*size*size)。')
    parser.add_argument('--fft-log-amplitude', action='store_true',
                        help='对幅度谱使用 log1p 压缩（默认关闭）。')
    parser.add_argument('--fft-no-log', dest='fft_log_amplitude', action='store_false')
    parser.set_defaults(fft_log_amplitude=True)
    parser.add_argument('--fft-shift', action='store_true',
                        help='是否做频谱中心化 (fftshift)。默认开启。')
    parser.add_argument('--fft-no-shift', dest='fft_shift', action='store_false')
    parser.set_defaults(fft_shift=True)
    parser.add_argument('--channel-aggregation', choices=['stack', 'mean', 'max'],
                        default='stack', help='通道聚合方式。stack=保留所有通道并展平。')
    parser.add_argument('--normalize-features', dest='normalize_features', action='store_true',
                        help='聚类前做 L2 归一化。')
    parser.add_argument('--no-normalize-features', dest='normalize_features', action='store_false')
    parser.set_defaults(normalize_features=True)
    parser.add_argument('--save-tokens', default=None,
                        help='可选，保存所有频域特征向量的路径 (.pth)。')
    parser.add_argument('--num-prototypes', type=int, default=16, help='聚类后的原型数量。')
    parser.add_argument('--sinkhorn-epochs', type=int, default=120)
    parser.add_argument('--sinkhorn-batch-size', type=int, default=512)
    parser.add_argument('--sinkhorn-queue-size', type=int, default=8192)
    parser.add_argument('--sinkhorn-momentum', type=float, default=0.02)
    parser.add_argument('--sinkhorn-iterations', type=int, default=5)
    parser.add_argument('--sinkhorn-epsilon', type=float, default=1e-2)
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


def _compute_frequency_vector(
    single_img: torch.Tensor,
    pool_size: int,
    log_amplitude: bool,
    do_shift: bool,
    channel_agg: str,
) -> torch.Tensor:
    """single_img: 形状为 (1, C, H, W)。"""
    if single_img.dim() != 4 or single_img.size(0) != 1:
        raise ValueError('single_img 必须是 (1, C, H, W) 张量。')
    freq_map = torch.fft.fft2(single_img, dim=(-2, -1))
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

    vector = magnitude.reshape(magnitude.size(0), -1).squeeze(0)
    return vector


@torch.no_grad()
def extract_frequency_prototypes(args: argparse.Namespace) -> Dict:
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

        for batch_idx in range(imgs.size(0)):
            if args.max_samples is not None and processed >= args.max_samples:
                break
            processed += 1
            if progress is not None:
                progress.update()

            single_img = imgs[batch_idx:batch_idx + 1]
            freq_vec = _compute_frequency_vector(
                single_img,
                pool_size=args.freq_pool_size,
                log_amplitude=args.fft_log_amplitude,
                do_shift=args.fft_shift,
                channel_agg=args.channel_aggregation,
            )
            feature_list.append(freq_vec.cpu().float())

    elapsed = time.time() - start_time

    if not feature_list:
        raise RuntimeError('未收集到任何频域特征，请检查数据集或配置。')

    features_tensor = torch.stack(feature_list, dim=0)
    total_images = features_tensor.size(0)
    feature_dim = features_tensor.size(1)

    # Sinkhorn 聚类（单一类别 id=0）
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
        'freq_pool_size': args.freq_pool_size,
        'fft_log_amplitude': args.fft_log_amplitude,
        'fft_shift': args.fft_shift,
        'channel_aggregation': args.channel_aggregation,
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

    output = {
        'metadata': metadata,
        'freq_proto': freq_proto,
        'freq_stats': {
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
        f"\nFrequency prototype statistics saved to {output_path}\n"
        f"Processed {processed} images in {elapsed:.2f}s (feature_dim={feature_dim})"
    )

    if args.save_tokens:
        token_payload = {
            'features': features_tensor,
            'metadata': {k: v for k, v in metadata.items() if k not in ['cluster_metrics', 'sinkhorn']}
        }
        torch.save(token_payload, args.save_tokens)
        print(f'Frequency feature tokens saved to {args.save_tokens}')

    return output


def main() -> None:
    args = parse_args()
    extract_frequency_prototypes(args)


if __name__ == '__main__':
    main()
