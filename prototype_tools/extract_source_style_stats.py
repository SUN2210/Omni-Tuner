#!/usr/bin/env python3
"""离线统计各 Backbone Stage 的均值/方差用于风格校正。

使用方式：
    python prototype_tools/extract_source_style_stats.py \
        --config <config.py> --checkpoint <best.pth> --out <style_stats.pth> \
        --device cuda:0 --batch-size 8 --stages patch_embed stage1_pre stage1_post ...

默认会同时统计每个 stage 在 LayerNorm **之前** (`stage{i}_pre`) 与
**之后** (`stage{i}_post`) 的特征分布，并保存到同一个文件中。
如需更细控制，可通过 `--stages` 指定任意子集。
"""

import argparse
import os
import sys
from typing import Dict, List, Optional

import torch
from mmcv import Config, DictAction
from mmcv.runner import load_checkpoint
from mmcv.utils import ProgressBar

# 保证可以直接以脚本方式运行
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(ROOT))

from mmdet.datasets import build_dataset, build_dataloader  # noqa: E402
from mmdet.models import build_detector  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='提取 backbone stage 的均值/方差以支持风格校正模块'
    )
    parser.add_argument('--config', required=True, help='mmdet 配置文件路径')
    parser.add_argument('--checkpoint', required=True, help='训练好的检测器权重路径')
    parser.add_argument('--out', required=True, help='输出的 style stats 文件 (.pth)')
    parser.add_argument('--device', default='cuda:0', help='用于特征提取的设备')
    parser.add_argument('--batch-size', type=int, default=8, help='特征提取时的 batch size')
    parser.add_argument('--workers', type=int, default=4, help='DataLoader workers 数')
    parser.add_argument('--max-samples', type=int, default=None, help='最多处理的样本数量')
    parser.add_argument('--stages', nargs='+', default=None,
                        help='需要统计的 stage 名称，例如 patch_embed stage1_pre stage1_post')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction,
                        help='以 key=value 形式覆盖配置文件参数')
    parser.add_argument('--no-progress', action='store_true', help='不显示进度条')
    return parser.parse_args()


def _init_model(cfg: Config, checkpoint: str, device: torch.device):
    model = build_detector(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    model.eval()
    load_checkpoint(model, checkpoint, map_location='cpu')
    model.to(device)
    return model


def _build_dataloader(cfg: Config, batch_size: int, workers: int):
    dataset = build_dataset(cfg.data.train)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=batch_size,
        workers_per_gpu=workers,
        dist=False,
        shuffle=False)
    return dataset, data_loader


def _update_stats(store: Dict[str, Dict[str, torch.Tensor]], stage: str, feat: torch.Tensor) -> None:
    feat = feat.detach()
    c = feat.size(1)
    if stage not in store:
        store[stage] = {
            'sum': torch.zeros(c, dtype=torch.float64),
            'sum_sq': torch.zeros(c, dtype=torch.float64),
            'count': 0,
        }
    entry = store[stage]
    sum_val = feat.sum(dim=(0, 2, 3)).to(torch.float64).cpu()
    sum_sq_val = (feat * feat).sum(dim=(0, 2, 3)).to(torch.float64).cpu()
    entry['sum'] += sum_val
    entry['sum_sq'] += sum_sq_val
    entry['count'] += feat.size(0) * feat.size(2) * feat.size(3)


def _finalize(store: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, Dict[str, torch.Tensor]]:
    style_stats: Dict[str, Dict[str, torch.Tensor]] = {}
    for stage, entry in store.items():
        total = max(int(entry['count']), 1)
        mean = entry['sum'] / total
        ex2 = entry['sum_sq'] / total
        var = torch.clamp(ex2 - mean * mean, min=0.0)
        std = torch.sqrt(var + 1e-12)
        style_stats[stage] = {
            'mean': mean.to(torch.float32),
            'std': std.to(torch.float32),
            'count': total,
        }
    return style_stats


def main() -> None:
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # ====== 调试专用，取消注释则只处理10张图 ======
    # args.max_samples = 10

    device = torch.device(args.device)
    model = _init_model(cfg, args.checkpoint, device)
    if getattr(model.backbone, 'style_adapter_enabled', False):
        print('[StyleStats] Detected active style adapter, temporarily disabling during extraction.')
        model.backbone.style_adapter_enabled = False
    dataset, data_loader = _build_dataloader(cfg, args.batch_size, args.workers)

    backbone_cfg = cfg.model.get('backbone', {})
    out_indices = backbone_cfg.get('out_indices', (0, 1, 2, 3))
    num_layers = getattr(model.backbone, 'num_layers', None)
    if num_layers is None:
        depths = backbone_cfg.get('depths')
        if depths is not None:
            num_layers = len(depths)
        else:
            num_layers = len(out_indices)
    all_stage_keys = [f'stage{i}' for i in range(int(num_layers))]
    default_stages: List[str] = ['patch_embed']
    for idx, stage in enumerate(all_stage_keys):
        default_stages.append(f'{stage}_pre')
        norm_attr = getattr(model.backbone, f'norm{idx}', None)
        if norm_attr is not None:
            default_stages.append(f'{stage}_post')

    stages = args.stages if args.stages is not None else default_stages
    stages = [stage.strip() for stage in stages]
    stages_set = set(stages)

    collect_patch = args.stages is None or 'patch_embed' in stages_set
    stage_filter = None
    if args.stages is not None:
        stage_filter = {stage for stage in stages_set if stage != 'patch_embed'}
    style_store: Dict[str, Dict[str, torch.Tensor]] = {}

    total_imgs = len(dataset)
    if args.max_samples is not None:
        total_imgs = min(total_imgs, args.max_samples)
    progress = None if args.no_progress else ProgressBar(total_imgs)

    processed = 0
    with torch.no_grad():
        for data in data_loader:
            imgs = data['img'].data[0].to(device, non_blocking=True)

            feature_dict = model.backbone.forward_style_features(
                imgs,
                stages=stage_filter if stage_filter else None,
                include_patch=collect_patch,
            )

            for stage_name, feat in feature_dict.items():
                if stage_name in stages_set:
                    _update_stats(style_store, stage_name, feat)

            batch_sz = imgs.size(0)
            processed += batch_sz
            if progress is not None:
                progress.update(batch_sz)
            if args.max_samples is not None and processed >= args.max_samples:
                break

    if progress is not None:
        print()

    style_stats = _finalize(style_store)
    stage_feature_spaces = {}
    for key in style_stats.keys():
        if key.endswith('_pre'):
            stage_feature_spaces[key] = 'pre_norm'
        elif key.endswith('_post'):
            stage_feature_spaces[key] = 'post_norm'
        elif key == 'patch_embed':
            stage_feature_spaces[key] = 'patch_embed'
        else:
            stage_feature_spaces[key] = 'unknown'

    metadata = {
        'config': os.path.abspath(args.config),
        'checkpoint': os.path.abspath(args.checkpoint),
        'stages': sorted(style_stats.keys()),
        'samples_processed': processed,
        'feature_space': 'pre_norm+post_norm',
    'stage_feature_spaces': stage_feature_spaces,
    'style_propagation_mode': getattr(model.backbone, 'style_propagation_mode', 'unknown'),
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save({'style_stats': style_stats, 'metadata': metadata}, args.out)
    print(f'[StyleStats] Saved {len(style_stats)} stages to {args.out}')


if __name__ == '__main__':
    main()
