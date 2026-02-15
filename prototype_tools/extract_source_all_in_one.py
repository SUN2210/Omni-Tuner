#!/usr/bin/env python3
"""Run style-statistics extraction and three prototype extractions in one sweep.

This utility iterates over the source training set a single time, sharing the
same detector backbone/dataloader to generate:

- LayerNorm pre/post style statistics for the backbone
- ROI-level class prototypes ("screen")
- Frequency-domain global prototypes
- Feature-space global prototypes

It consolidates the logic from the individual scripts under
prototype_tools/ into a single pipeline so that images are only forwarded
through the network once per task group.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from mmcv import Config, DictAction
from mmcv.utils import ProgressBar

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(ROOT))

from mmdet.core import bbox2roi  # noqa: E402

from prototype_tools import extract_source_style_stats as style_mod  # noqa: E402
from prototype_tools import extract_source_prototypes_screen as screen_mod  # noqa: E402
from prototype_tools import extract_source_prototypes_global as global_mod  # noqa: E402
from prototype_tools import extract_source_prototypes_backbone_frequency as backbone_freq_mod  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Extract style stats and prototypes with a single dataloader pass.'
    )
    parser.add_argument('--config', required=True, help='Path to mmdet config file.')
    parser.add_argument('--checkpoint', required=True, help='Path to trained detector weights (.pth).')
    parser.add_argument('--device', default='cuda:0', help='Device to use for feature extraction.')
    parser.add_argument('--batch-size', type=int, default=16, help='Samples per GPU while iterating.')
    parser.add_argument('--workers', type=int, default=8, help='Number of dataloader workers.')
    parser.add_argument('--max-samples', type=int, default=None, help='Cap the number of processed images.')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction, help='Override config settings key=value.')
    parser.add_argument('--no-progress', action='store_true', help='Disable tqdm-like progress bar output.')

    parser.add_argument('--style-out', required=True, help='Output file (.pth) for style statistics.')
    parser.add_argument('--style-stages', nargs='+', default=None,
                        help='Stage names to collect (default mirrors standalone script).')

    parser.add_argument('--screen-out', required=True, help='Output file (.pth) for ROI prototypes.')
    parser.add_argument('--screen-save-tokens', default=None,
                        help='Optional file to dump ROI token features.')
    parser.add_argument('--screen-num-prototypes', type=int, default=3, help='Number of prototypes per class.')
    parser.add_argument('--screen-normalize-features', dest='screen_normalize_features', action='store_true')
    parser.add_argument('--screen-no-normalize-features', dest='screen_normalize_features', action='store_false')
    parser.set_defaults(screen_normalize_features=True)
    parser.add_argument('--screen-sinkhorn-epochs', type=int, default=100)
    parser.add_argument('--screen-sinkhorn-batch-size', type=int, default=512)
    parser.add_argument('--screen-sinkhorn-queue-size', type=int, default=8192)
    parser.add_argument('--screen-sinkhorn-momentum', type=float, default=0.02)
    parser.add_argument('--screen-sinkhorn-iterations', type=int, default=5)
    parser.add_argument('--screen-sinkhorn-epsilon', type=float, default=1e-2)
    parser.add_argument('--screen-enable-filtering', dest='screen_enable_filtering', action='store_true')
    parser.add_argument('--screen-disable-filtering', dest='screen_enable_filtering', action='store_false')
    parser.set_defaults(screen_enable_filtering=True)
    parser.add_argument('--screen-filter-outliers-ratio', type=float, default=0.1)
    parser.add_argument('--screen-filter-knn-ratio', type=float, default=0.05)
    parser.add_argument('--screen-filter-activation-ratio', type=float, default=0.05)
    parser.add_argument('--screen-l2-normalize', action='store_true', help='Store l2-normalized prototype copy.')
    parser.add_argument('--screen-no-visualize', action='store_true',
                        help='Skip visualization/report generation for ROI prototypes.')

    parser.add_argument('--freq-out', required=True, help='Output file (.pth) for frequency prototypes.')
    parser.add_argument('--freq-save-tokens', default=None,
                        help='Optional file to dump raw frequency vectors.')
    parser.add_argument('--freq-feature-stage', choices=['c2', 'c3', 'c4', 'c5', 'last'], default='c5')
    parser.add_argument('--freq-pool-size', type=int, default=24)
    parser.add_argument('--freq-fft-log-amplitude', dest='freq_fft_log_amplitude', action='store_true')
    parser.add_argument('--freq-fft-no-log', dest='freq_fft_log_amplitude', action='store_false')
    parser.set_defaults(freq_fft_log_amplitude=True)
    parser.add_argument('--freq-fft-shift', dest='freq_fft_shift', action='store_true')
    parser.add_argument('--freq-fft-no-shift', dest='freq_fft_shift', action='store_false')
    parser.set_defaults(freq_fft_shift=True)
    parser.add_argument('--freq-channel-aggregation', choices=['stack', 'mean', 'max'], default='mean')
    parser.add_argument('--freq-channel-subsample', type=int, default=None,
                        help='Optional channel subsample when aggregation=stack; keeps first N channels.')
    parser.add_argument('--freq-num-prototypes', type=int, default=16)
    parser.add_argument('--freq-normalize-features', dest='freq_normalize_features', action='store_true')
    parser.add_argument('--freq-no-normalize-features', dest='freq_normalize_features', action='store_false')
    parser.set_defaults(freq_normalize_features=True)
    parser.add_argument('--freq-sinkhorn-epochs', type=int, default=120)
    parser.add_argument('--freq-sinkhorn-batch-size', type=int, default=512)
    parser.add_argument('--freq-sinkhorn-queue-size', type=int, default=8192)
    parser.add_argument('--freq-sinkhorn-momentum', type=float, default=0.02)
    parser.add_argument('--freq-sinkhorn-iterations', type=int, default=5)
    parser.add_argument('--freq-sinkhorn-epsilon', type=float, default=1e-2)

    parser.add_argument('--global-out', required=True, help='Output file (.pth) for global feature prototypes.')
    parser.add_argument('--global-save-tokens', default=None,
                        help='Optional file to dump pooled global features.')
    parser.add_argument('--global-feature-source', choices=['p5', 'c4'], default='c4')
    parser.add_argument('--global-pool-type', choices=['avg', 'avgmax'], default='avg')
    parser.add_argument('--global-normalize-features', dest='global_normalize_features', action='store_true')
    parser.add_argument('--global-no-normalize-features', dest='global_normalize_features', action='store_false')
    parser.set_defaults(global_normalize_features=True)
    parser.add_argument('--global-l2-normalize', action='store_true')
    parser.add_argument('--global-num-prototypes', type=int, default=12)
    parser.add_argument('--global-sinkhorn-epochs', type=int, default=100)
    parser.add_argument('--global-sinkhorn-batch-size', type=int, default=512)
    parser.add_argument('--global-sinkhorn-queue-size', type=int, default=8192)
    parser.add_argument('--global-sinkhorn-momentum', type=float, default=0.02)
    parser.add_argument('--global-sinkhorn-iterations', type=int, default=5)
    parser.add_argument('--global-sinkhorn-epsilon', type=float, default=1e-2)
    parser.add_argument('--global-no-visualize', action='store_true',
                        help='Skip visualization/report generation for global prototypes.')

    parser.add_argument('--no-visualize', action='store_true',
                        help='Shortcut to disable all prototype visualizations/reports.')
    return parser.parse_args()


def _stage_sort_key(name: str) -> Tuple[int, int]:
    idx = -1
    if name.startswith('stage'):
        digits = []
        for ch in name[5:]:
            if ch.isdigit():
                digits.append(ch)
            else:
                break
        if digits:
            idx = int(''.join(digits))
    return idx, 0 if name.endswith('_pre') else 1


def _prepare_style_stages(cfg: Config, model, requested: Optional[Iterable[str]]) -> Tuple[List[str], set, Optional[List[str]], bool]:
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
    if requested is None:
        stages = default_stages
    else:
        stages = [item.strip() for item in requested]
    stage_set = set(stages)
    collect_patch = 'patch_embed' in stage_set
    stage_filter = [s for s in stages if s != 'patch_embed']
    return stages, stage_set, stage_filter if stage_filter else None, collect_patch


def _ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)


def _to_list(data, length: int):
    if isinstance(data, (list, tuple)):
        return list(data)[:length]
    if torch.is_tensor(data):
        return [data[i] for i in range(length)]
    raise TypeError(f'Unsupported container type: {type(data)}')


def main() -> None:
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    global_mod._maybe_import_custom_modules(cfg)

    device = torch.device(args.device)
    model = style_mod._init_model(cfg, args.checkpoint, device)
    if getattr(model.backbone, 'style_adapter_enabled', False):
        print('[Combined] Temporarily disabling style adapter during extraction.')
        model.backbone.style_adapter_enabled = False

    dataset, data_loader = style_mod._build_dataloader(cfg, args.batch_size, args.workers)

    stages, stage_set, stage_filter, collect_patch = _prepare_style_stages(cfg, model, args.style_stages)
    stage_filter_arg = stage_filter if stage_filter else None

    total_imgs = len(dataset)
    if args.max_samples is not None:
        total_imgs = min(total_imgs, args.max_samples)

    progress = None if args.no_progress else ProgressBar(total_imgs)

    style_store: Dict[str, Dict[str, torch.Tensor]] = {}
    roi_feature_store: Dict[int, List[torch.Tensor]] = defaultdict(list)
    frequency_features: List[torch.Tensor] = []
    global_features: List[torch.Tensor] = []

    total_instances = 0
    processed = 0
    loop_start = time.time()

    with torch.no_grad():
        for data in data_loader:
            if args.max_samples is not None and processed >= args.max_samples:
                break

            imgs = screen_mod._unwrap_data_container(data['img'])
            if isinstance(imgs, list):
                imgs = torch.stack(imgs, dim=0)
            imgs = imgs.to(device)

            batch_size = imgs.size(0)
            if args.max_samples is not None:
                remaining = args.max_samples - processed
                if remaining <= 0:
                    break
                if remaining < batch_size:
                    imgs = imgs[:remaining]
                    batch_size = imgs.size(0)

            gt_bboxes_batch = screen_mod._unwrap_data_container(data['gt_bboxes'])
            gt_labels_batch = screen_mod._unwrap_data_container(data['gt_labels'])
            gt_bboxes_list = _to_list(gt_bboxes_batch, batch_size)
            gt_labels_list = _to_list(gt_labels_batch, batch_size)

            feature_dict = model.backbone.forward_style_features(
                imgs,
                stages=stage_filter_arg,
                include_patch=collect_patch,
            )

            for stage_name, feat in feature_dict.items():
                if stage_name in stage_set:
                    style_mod._update_stats(style_store, stage_name, feat)

            last_stage_tensor: Optional[torch.Tensor] = None
            if args.global_feature_source == 'c4':
                post_candidates = [
                    (name, feature_dict[name])
                    for name in feature_dict
                    if name.endswith('_post') and name.startswith('stage') and isinstance(feature_dict[name], torch.Tensor)
                ]
                if post_candidates:
                    post_candidates.sort(key=lambda item: _stage_sort_key(item[0]))
                    last_stage_tensor = post_candidates[-1][1]

            feats = model.extract_feat(imgs)

            rois_list = []
            roi_labels = []
            for img_idx in range(batch_size):
                gt_bboxes = torch.as_tensor(
                    gt_bboxes_list[img_idx], device=device, dtype=torch.float32
                )
                gt_labels = torch.as_tensor(
                    gt_labels_list[img_idx], device=device, dtype=torch.long
                )
                if gt_bboxes.numel() == 0:
                    continue

                rois = bbox2roi([gt_bboxes])
                rois[:, 0] = img_idx
                rois_list.append(rois)
                roi_labels.append(gt_labels)

            if rois_list:
                rois_cat = torch.cat(rois_list, dim=0)
                roi_feats = screen_mod._extract_roi_features(model, feats, rois_cat, device)
                pooled = roi_feats.mean(dim=[2, 3])
                labels_cat = torch.cat(roi_labels, dim=0)
                pooled_cpu = pooled.cpu().float()
                labels_cpu = labels_cat.cpu().long()
                for feat_vec, cls in zip(pooled_cpu, labels_cpu):
                    cls_id = int(cls.item())
                    roi_feature_store[cls_id].append(feat_vec)
                    total_instances += 1

            backbone_outputs = model.backbone(imgs)
            stage_feat = backbone_freq_mod._select_stage(backbone_outputs, args.freq_feature_stage)
            if isinstance(stage_feat, (list, tuple)):
                stage_feat = stage_feat[-1]
            stage_feat = stage_feat.to(device)

            for img_idx in range(batch_size):
                single_feat = stage_feat[img_idx:img_idx + 1]
                freq_vec = backbone_freq_mod._freq_vector_from_feature(
                    single_feat,
                    pool_size=args.freq_pool_size,
                    log_amplitude=args.freq_fft_log_amplitude,
                    do_shift=args.freq_fft_shift,
                    channel_agg=args.freq_channel_aggregation,
                    channel_subsample=args.freq_channel_subsample,
                )
                frequency_features.append(freq_vec.cpu().float())

            if args.global_feature_source == 'c4' and last_stage_tensor is not None:
                pooled_global = global_mod._global_pool(last_stage_tensor, args.global_pool_type)
            else:
                if isinstance(feats, torch.Tensor):
                    feature_map = feats
                elif isinstance(feats, (list, tuple)):
                    feature_map = feats[-1]
                elif isinstance(feats, dict):
                    feature_map = feats[list(feats.keys())[-1]]
                else:
                    raise TypeError(f'Unsupported feature type returned by model.extract_feat: {type(feats)}')
                pooled_global = global_mod._global_pool(feature_map, args.global_pool_type)

            for row in pooled_global.cpu().float():
                global_features.append(row)

            processed += batch_size
            if progress is not None:
                progress.update(batch_size)

            if args.max_samples is not None and processed >= args.max_samples:
                break

    elapsed = time.time() - loop_start
    if progress is not None:
        print()

    if processed == 0:
        raise RuntimeError('No samples were processed. Please check dataloader configuration.')

    # ===== Style statistics =====
    style_stats = style_mod._finalize(style_store)
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

    style_metadata = {
        'config': os.path.abspath(args.config),
        'checkpoint': os.path.abspath(args.checkpoint),
        'stages': sorted(style_stats.keys(), key=_stage_sort_key),
        'samples_processed': processed,
        'feature_space': 'pre_norm+post_norm',
        'stage_feature_spaces': stage_feature_spaces,
        'style_propagation_mode': getattr(model.backbone, 'style_propagation_mode', 'unknown'),
    }
    _ensure_dir(args.style_out)
    torch.save({'style_stats': style_stats, 'metadata': style_metadata}, args.style_out)
    print(f'[Combined] Saved style statistics to {args.style_out}')

    # ===== ROI prototypes =====
    if not roi_feature_store:
        raise RuntimeError('No ROI features were collected. Unable to compute prototypes.')

    stacked_roi_features: Dict[int, torch.Tensor] = {}
    for cls_id, features in roi_feature_store.items():
        if features:
            stacked_roi_features[int(cls_id)] = torch.stack(features, dim=0)
    if not stacked_roi_features:
        raise RuntimeError('ROI feature store was populated but empty after stacking.')

    per_class_counts = {cls_id: int(tensor.size(0)) for cls_id, tensor in stacked_roi_features.items()}
    feature_dim = next(iter(stacked_roi_features.values())).size(1)

    filtered_instance_counts = None
    instances_after_filtering = sum(per_class_counts.values())

    if args.screen_enable_filtering:
        stacked_roi_features = screen_mod.apply_feature_filtering(
            stacked_roi_features,
            filter_outliers_ratio=args.screen_filter_outliers_ratio,
            filter_knn_ratio=args.screen_filter_knn_ratio,
            filter_activation_ratio=args.screen_filter_activation_ratio,
            verbose=not (args.no_progress or args.screen_no_visualize or args.no_visualize),
        )
        filtered_instance_counts = {cls_id: int(tensor.size(0)) for cls_id, tensor in stacked_roi_features.items()}
        instances_after_filtering = sum(filtered_instance_counts.values())

    if args.screen_save_tokens:
        token_payload = {
            'features': {cls_id: tensor for cls_id, tensor in stacked_roi_features.items()},
            'class_names': getattr(dataset, 'CLASSES', None),
            'metadata': {
                'config': args.config,
                'checkpoint': args.checkpoint,
                'feature_dim': feature_dim,
                'normalize_features': args.screen_normalize_features,
                'feature_filtering_enabled': args.screen_enable_filtering,
                'instances_after_filtering': instances_after_filtering if args.screen_enable_filtering else None,
            }
        }
        _ensure_dir(args.screen_save_tokens)
        torch.save(token_payload, args.screen_save_tokens)

    prototypes_dict, screen_metrics = screen_mod.compute_prototypes_sinkhorn(
        stacked_roi_features,
        num_prototypes=args.screen_num_prototypes,
        normalize_features=args.screen_normalize_features,
        device=device,
        epochs=args.screen_sinkhorn_epochs,
        batch_size=args.screen_sinkhorn_batch_size,
        queue_size=args.screen_sinkhorn_queue_size,
        momentum=args.screen_sinkhorn_momentum,
        iterations=args.screen_sinkhorn_iterations,
        epsilon=args.screen_sinkhorn_epsilon,
    )

    for entry in prototypes_dict.values():
        entry['method'] = 'sinkhorn'
        if args.screen_l2_normalize:
            entry['centers_l2'] = F.normalize(entry['centers'], dim=1)

    roi_metadata = {
        'config': args.config,
        'checkpoint': args.checkpoint,
        'device': str(device),
        'samples_processed': processed,
        'elapsed_seconds': elapsed,
        'class_names': getattr(dataset, 'CLASSES', None),
        'feature_dim': int(feature_dim),
        'cluster_method': 'sinkhorn',
        'num_prototypes': args.screen_num_prototypes,
        'normalize_features': args.screen_normalize_features,
        'l2_normalized_copy': args.screen_l2_normalize,
        'total_instances': sum(per_class_counts.values()),
        'per_class_instance_counts': per_class_counts,
        'cluster_metrics': screen_metrics,
        'tokens_saved_to': args.screen_save_tokens,
        'feature_filtering_enabled': args.screen_enable_filtering,
        'instances_after_filtering': instances_after_filtering if args.screen_enable_filtering else None,
        'filtered_instance_counts': filtered_instance_counts if args.screen_enable_filtering else None,
    }

    roi_output = {
        'metadata': roi_metadata,
        'prototypes': prototypes_dict,
    }

    _ensure_dir(args.screen_out)
    torch.save(roi_output, args.screen_out)
    print(f'[Combined] Saved ROI prototypes to {args.screen_out} (instances={total_instances})')

    # Optionally reuse visualization/report helpers when enabled
    should_visualize_screen = not (args.no_visualize or args.screen_no_visualize)
    if should_visualize_screen:
        output_dir = os.path.dirname(os.path.abspath(args.screen_out)) or '.'
        try:
            screen_mod._save_prototypes_report(output_dir, prototypes_dict, roi_metadata)
            screen_mod._visualize_tokens_and_prototypes(
                output_dir,
                {cls: tensor for cls, tensor in stacked_roi_features.items()},
                prototypes_dict,
                roi_metadata,
                max_tokens=5000,
                alpha=0.55,
                seed=0,
            )
        except Exception as exc:  # pragma: no cover
            print(f'[Combined][screen] Visualization/report skipped due to error: {exc}')

    # ===== Frequency prototypes =====
    if not frequency_features:
        raise RuntimeError('No frequency features were collected.')

    freq_tensor = torch.stack(frequency_features, dim=0)
    freq_feature_dim = freq_tensor.size(1)
    freq_features_for_cluster = F.normalize(freq_tensor, dim=1) if args.freq_normalize_features else freq_tensor
    freq_features_per_class = {0: freq_features_for_cluster}

    freq_prototypes, freq_metrics = global_mod.compute_prototypes_sinkhorn(
        freq_features_per_class,
        num_prototypes=args.freq_num_prototypes,
        normalize_features=args.freq_normalize_features,
        device=device,
        epochs=args.freq_sinkhorn_epochs,
        batch_size=args.freq_sinkhorn_batch_size,
        queue_size=args.freq_sinkhorn_queue_size,
        momentum=args.freq_sinkhorn_momentum,
        iterations=args.freq_sinkhorn_iterations,
        epsilon=args.freq_sinkhorn_epsilon,
    )

    freq_entry = freq_prototypes[0]
    freq_entry['method'] = 'sinkhorn'
    mean_vec = freq_tensor.mean(dim=0)
    var_vec = freq_tensor.var(dim=0, unbiased=False)

    freq_metadata = {
        'config': args.config,
        'checkpoint': args.checkpoint,
        'device': str(device),
        'samples_processed': processed,
        'elapsed_seconds': elapsed,
        'feature_stage': args.freq_feature_stage,
        'freq_pool_size': args.freq_pool_size,
        'fft_log_amplitude': args.freq_fft_log_amplitude,
        'fft_shift': args.freq_fft_shift,
        'channel_aggregation': args.freq_channel_aggregation,
        'channel_subsample': args.freq_channel_subsample,
        'normalize_features': args.freq_normalize_features,
        'feature_dim': int(freq_feature_dim),
        'num_prototypes': args.freq_num_prototypes,
        'cluster_metrics': freq_metrics,
        'total_images': freq_tensor.size(0),
        'sinkhorn': {
            'epochs': args.freq_sinkhorn_epochs,
            'batch_size': args.freq_sinkhorn_batch_size,
            'queue_size': args.freq_sinkhorn_queue_size,
            'momentum': args.freq_sinkhorn_momentum,
            'iterations': args.freq_sinkhorn_iterations,
            'epsilon': args.freq_sinkhorn_epsilon,
        },
        'tokens_saved_to': args.freq_save_tokens,
        'input_source': 'backbone',
    }

    freq_output = {
        'metadata': freq_metadata,
        'freq_proto': freq_entry,
        'freq_stats': {
            'mean': mean_vec,
            'var': var_vec,
        },
    }

    _ensure_dir(args.freq_out)
    torch.save(freq_output, args.freq_out)
    print(f'[Combined] Saved frequency prototypes to {args.freq_out}')

    if args.freq_save_tokens:
        _ensure_dir(args.freq_save_tokens)
        torch.save({'features': freq_tensor, 'metadata': freq_metadata}, args.freq_save_tokens)

    # ===== Global prototypes =====
    if not global_features:
        raise RuntimeError('No global pooled features were collected.')

    global_tensor = torch.stack(global_features, dim=0)
    global_feature_dim = global_tensor.size(1)
    global_features_for_cluster = (
        F.normalize(global_tensor, dim=1) if args.global_normalize_features else global_tensor
    )
    global_features_per_class = {0: global_features_for_cluster}

    global_prototypes, global_metrics = global_mod.compute_prototypes_sinkhorn(
        global_features_per_class,
        num_prototypes=args.global_num_prototypes,
        normalize_features=args.global_normalize_features,
        device=device,
        epochs=args.global_sinkhorn_epochs,
        batch_size=args.global_sinkhorn_batch_size,
        queue_size=args.global_sinkhorn_queue_size,
        momentum=args.global_sinkhorn_momentum,
        iterations=args.global_sinkhorn_iterations,
        epsilon=args.global_sinkhorn_epsilon,
    )

    global_entry = global_prototypes[0]
    global_entry['method'] = 'sinkhorn'
    if args.global_l2_normalize:
        global_entry['centers_l2'] = F.normalize(global_entry['centers'], dim=1)

    global_mean = global_tensor.mean(dim=0)
    global_var = global_tensor.var(dim=0, unbiased=False)

    global_metadata = {
        'config': args.config,
        'checkpoint': args.checkpoint,
        'device': str(device),
        'samples_processed': processed,
        'elapsed_seconds': elapsed,
        'feature_source': args.global_feature_source,
        'pool_type': args.global_pool_type,
        'normalize_features': args.global_normalize_features,
        'l2_normalized_copy': args.global_l2_normalize,
        'feature_dim': int(global_feature_dim),
        'num_prototypes': args.global_num_prototypes,
        'cluster_method': 'sinkhorn',
        'cluster_metrics': global_metrics,
        'total_images': global_tensor.size(0),
        'sinkhorn': {
            'epochs': args.global_sinkhorn_epochs,
            'batch_size': args.global_sinkhorn_batch_size,
            'queue_size': args.global_sinkhorn_queue_size,
            'momentum': args.global_sinkhorn_momentum,
            'iterations': args.global_sinkhorn_iterations,
            'epsilon': args.global_sinkhorn_epsilon,
        },
        'tokens_saved_to': args.global_save_tokens,
    }

    global_output = {
        'metadata': global_metadata,
        'global_proto': global_entry,
        'global_stats': {
            'mean': global_mean,
            'var': global_var,
        },
    }

    _ensure_dir(args.global_out)
    torch.save(global_output, args.global_out)
    print(f'[Combined] Saved global prototypes to {args.global_out}')

    if args.global_save_tokens:
        _ensure_dir(args.global_save_tokens)
        torch.save({'features': global_tensor, 'metadata': global_metadata}, args.global_save_tokens)

    should_visualize_global = not (args.no_visualize or args.global_no_visualize)
    if should_visualize_global:
        output_dir = os.path.dirname(os.path.abspath(args.global_out)) or '.'
        try:
            if hasattr(global_mod, '_visualize_global_features_and_prototypes'):
                global_mod._visualize_global_features_and_prototypes(
                    output_dir,
                    global_tensor,
                    global_prototypes,
                    global_metadata,
                    max_tokens=min(4000, global_tensor.size(0)),
                    alpha=0.55,
                    seed=0,
                )
            if hasattr(global_mod, '_visualize_prototype_distance_matrix'):
                global_mod._visualize_prototype_distance_matrix(output_dir, global_prototypes)
            if hasattr(global_mod, '_save_prototypes_report'):
                global_mod._save_prototypes_report(output_dir, global_prototypes, global_metadata)
        except Exception as exc:  # pragma: no cover
            print(f'[Combined][global] Visualization skipped due to error: {exc}')

    print('[Combined] Completed in {:.2f} seconds over {} images.' .format(elapsed, processed))


if __name__ == '__main__':
    main()
