import argparse
import copy
import os
import json # 导入 json 模块


# 设置环境变量，指定使用的GPU
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import os.path as osp
import time
import warnings
import sys

# 添加项目路径到 sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, '.')

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist
from mmcv.utils import get_git_hash

from mmdet import __version__
from mmdet.apis import set_random_seed, train_detector
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.utils import collect_env, get_root_logger

from prototype_tools.prototype_adv_manager_Prototype1001 import PrototypeAdvManagerPrototype1001
from prototype_tools.global_prototype_adv_manager import GlobalPrototypeAdvManager
from prototype_tools.frequency_prototype_adv_manager import FrequencyPrototypeAdvManager

def _count_module_parameters(module):
    """Return total and trainable parameter counts for a module."""
    total = 0
    trainable = 0
    for param in module.parameters():
        param_count = param.numel()
        total += param_count
        if param.requires_grad:
            trainable += param_count
    return total, trainable


def summarize_model_architecture(module, max_depth=2, max_children=20):
    """Create a readable hierarchical summary of the model architecture."""

    def _summarize(current_module, module_name, depth):
        indent = '  ' * depth
        total_params, trainable_params = _count_module_parameters(current_module)
        child_items = list(current_module.named_children())
        lines = [
            f"{indent}- {module_name}: {current_module.__class__.__name__} "
            f"| params={total_params:,} (trainable={trainable_params:,})"
        ]
        if depth >= max_depth or not child_items:
            return lines

        for idx, (child_name, child_module) in enumerate(child_items):
            if idx >= max_children:
                remaining = len(child_items) - idx
                lines.append(f"{indent}  ... {remaining} more submodules omitted ...")
                break
            lines.extend(_summarize(child_module, child_name or '<unnamed>', depth + 1))
        return lines

    return _summarize(module, module.__class__.__name__, 0)



def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    # 支持 positional argument 方式传递 config 文件路径
    parser.add_argument('config', nargs='?', default='./work_dirs/Omni_Tuner_configs/swin-l_dota3c_fs/dota3c_fs_retinanet_swin_large_5x_smallbackbone4.py', help='train config file path (positional or --config)')
    parser.add_argument('--config', dest='config_kw', help=argparse.SUPPRESS)
    
    
    
    # parser.add_argument('--config', default='./work_dirs/Omni_Tuner_configs/swin-l_hrrsd1c/hrrsd1c_retinanet_swin_large_1x_myvpt3.py', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file (deprecate), '
        'change to --cfg-options instead.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--prototype-file', default=None,
                        help='Path to source-domain prototype statistics (.pth).')
    parser.add_argument('--prototype-align-weight', type=float, default=None,
                        help='Override alignment loss weight defined in config.')
    parser.add_argument('--prototype-adv-weight', type=float, default=None,
                        help='Override adversarial loss weight defined in config.')
    parser.add_argument('--prototype-temperature', type=float, default=None,
                        help='Override prototype sampling temperature (positive enables soft sampling).')
    parser.add_argument('--prototype-momentum', type=float, default=None,
                        help='Override prototype bank momentum (0 disables online updates).')
    parser.add_argument('--global-prototype-file', default=None,
                        help='Path to global image-level prototype statistics (.pth).')
    parser.add_argument('--global-proto-align-weight', type=float, default=None,
                        help='Override global alignment loss weight.')
    parser.add_argument('--global-proto-adv-weight', type=float, default=None,
                        help='Override global adversarial loss weight.')
    parser.add_argument('--global-proto-feature-source', choices=['c4', 'p5'], default=None,
                        help='Override feature source used for global prototypes (c4 or p5).')
    parser.add_argument('--global-proto-pool-type', choices=['avg', 'avgmax'], default=None,
                        help='Override pooling type for global feature aggregation.')
    parser.add_argument('--frequency-prototype-file', default=None,
                        help='Path to frequency-domain prototype statistics (.pth).')
    parser.add_argument('--frequency-proto-align-weight', type=float, default=None,
                        help='Override frequency alignment loss weight.')
    parser.add_argument('--frequency-proto-adv-weight', type=float, default=None,
                        help='Override frequency adversarial loss weight.')
    parser.add_argument('--frequency-proto-pool-size', type=int, default=None,
                        help='Override frequency pooling size (default根据原型统计存档)。')
    parser.add_argument('--frequency-proto-log-amplitude', type=str, choices=['auto', 'on', 'off'], default='auto',
                        help='Control log-amplitude transform; auto uses prototype metadata.')
    parser.add_argument('--frequency-proto-shift', type=str, choices=['auto', 'on', 'off'], default='auto',
                        help='Control fftshift option; auto uses prototype metadata.')
    parser.add_argument('--frequency-proto-channel-aggregation', choices=['stack', 'mean', 'max', 'auto'], default='auto',
                        help='Channel aggregation for FFT features; auto uses prototype metadata.')
    parser.add_argument('--frequency-proto-input-source', choices=['image', 'backbone', 'auto'], default='auto',
                        help='Control whether frequency manager consumes raw images or backbone features.')
    parser.add_argument('--frequency-proto-feature-stage', default=None,
                        help='Override backbone stage used for frequency tokens (e.g., c5).')
    parser.add_argument('--frequency-proto-channel-subsample', type=int, default=None,
                        help='Override channel subsample count when using stack aggregation.')
    parser.add_argument('--style-stats', default=None,
                        help='Path to style adapter statistics (.pth) for the backbone.')
    parser.add_argument('--style-momentum', type=float, default=None,
                        help='Override style adapter momentum (0~1).')
    parser.add_argument('--style-apply-to', type=str, default=None,
                        help='Comma-separated stage keys to enable style adaptation (e.g., "stage3" or "patch_embed,stage3").')
    parser.add_argument('--style-mix-alpha', type=str, default=None,
                        help='Blend coefficient between adapted and original features (0~1) or "x" for learnable.')
    parser.add_argument('--style-mix-alpha-init', type=float, default=None,
                        help='Initial value when --style-mix-alpha is set to "x".')
    parser.add_argument('--style-mix-alpha-scope', type=str, default=None,
                        help='Control whether mix_alpha is shared or per-stage ("shared" or "per-stage").')
    parser.add_argument('--style-mix-alpha-lr-map', type=str, default=None,
                        help='Comma-separated stage:lr pairs to set per-stage mix_alpha LR multipliers when scope is per-stage.')
    parser.add_argument('--style-propagation', type=str, choices=['full', 'fpn_only', 'none'], default=None,
                        help='Control how style statistics propagate through the backbone.')

    args = parser.parse_args()
    # 如果通过 --config 传递，则覆盖 positional
    if getattr(args, 'config_kw', None) is not None:
        args.config = args.config_kw
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.cfg_options:
        raise ValueError(
            '--options and --cfg-options cannot be both '
            'specified, --options is deprecated in favor of --cfg-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --cfg-options')
        args.cfg_options = args.options

    return args


def main():
    args = parse_args()

    # ---------for quark-----------
    device_id = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(device_id)
    # ---------for quark-----------

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    if any(v is not None for v in [
        args.style_stats,
        args.style_momentum,
        args.style_apply_to,
        args.style_mix_alpha,
        args.style_mix_alpha_init,
        args.style_mix_alpha_scope,
        args.style_mix_alpha_lr_map,
        args.style_propagation,
    ]):
        backbone_cfg = cfg.model.get('backbone', None)
        if backbone_cfg is None:
            raise ValueError('Config missing model.backbone for style adapter override')
        style_cfg = backbone_cfg.setdefault('style_adapter', {})
        style_cfg.setdefault('enabled', True)
        if args.style_stats is not None:
            style_cfg['stats_file'] = args.style_stats
        if args.style_momentum is not None:
            style_cfg['momentum'] = args.style_momentum
        if args.style_apply_to is not None:
            stages = [s.strip() for s in args.style_apply_to.split(',') if s.strip()]
            style_cfg['apply_to'] = stages
        if args.style_mix_alpha is not None:
            mix_alpha_value = args.style_mix_alpha.strip()
            if mix_alpha_value.lower() == 'x':
                style_cfg['mix_alpha'] = 'x'
                if args.style_mix_alpha_init is not None:
                    style_cfg['mix_alpha_init'] = args.style_mix_alpha_init
            else:
                try:
                    style_cfg['mix_alpha'] = float(mix_alpha_value)
                except ValueError as exc:
                    raise ValueError(f'Invalid --style-mix-alpha value: {args.style_mix_alpha}') from exc
        if args.style_mix_alpha_scope is not None:
            scope_token = args.style_mix_alpha_scope.strip().lower()
            scope_token = scope_token.replace('-', '_')
            if scope_token not in {'shared', 'per_stage'}:
                raise ValueError(
                    '--style-mix-alpha-scope must be "shared" or "per-stage"'
                )
            style_cfg['mix_alpha_scope'] = scope_token
        if args.style_mix_alpha_lr_map is not None:
            lr_map_entries = {}
            raw_items = [item.strip() for item in args.style_mix_alpha_lr_map.split(',') if item.strip()]
            if not raw_items:
                raise ValueError('--style-mix-alpha-lr-map provided but no valid stage:lr entries found')
            for entry in raw_items:
                if ':' not in entry:
                    raise ValueError(f'Invalid style mix_alpha lr map entry "{entry}" (expected stage:lr)')
                stage_token, lr_token = entry.split(':', 1)
                stage_key = stage_token.strip()
                if not stage_key:
                    raise ValueError(f'Empty stage key in lr map entry "{entry}"')
                try:
                    lr_value = float(lr_token.strip())
                except ValueError as exc:
                    raise ValueError(f'Invalid lr multiplier "{lr_token}" in entry "{entry}"') from exc
                lr_map_entries[stage_key] = lr_value
            style_cfg['mix_alpha_lr_mult_map'] = lr_map_entries
        if args.style_propagation is not None:
            style_cfg['propagation'] = args.style_propagation

    backbone_cfg_ref = cfg.model.get('backbone', None)
    mix_alpha_lr_mult = None
    mix_alpha_lr_mult_map = {}
    if backbone_cfg_ref is not None:
        style_cfg_ref = backbone_cfg_ref.get('style_adapter', None)
        if style_cfg_ref is not None:
            raw_lr_mult = style_cfg_ref.get('mix_alpha_lr_mult', None)
            if raw_lr_mult is not None:
                try:
                    mix_alpha_lr_mult = float(raw_lr_mult)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f'Invalid mix_alpha_lr_mult value: {raw_lr_mult}') from exc
                style_cfg_ref['mix_alpha_lr_mult'] = mix_alpha_lr_mult
            raw_lr_mult_map = style_cfg_ref.get('mix_alpha_lr_mult_map', None)
            if raw_lr_mult_map:
                normalized_map = {}
                if isinstance(raw_lr_mult_map, dict):
                    items = raw_lr_mult_map.items()
                else:
                    raise ValueError('mix_alpha_lr_mult_map must be a dict mapping stage keys to lr multipliers')
                for stage_key, raw_val in items:
                    if stage_key is None:
                        continue
                    stage_key_str = str(stage_key).strip()
                    if not stage_key_str:
                        continue
                    try:
                        normalized_map[stage_key_str] = float(raw_val)
                    except (TypeError, ValueError) as exc:
                        raise ValueError(f'Invalid lr multiplier for stage {stage_key}: {raw_val}') from exc
                mix_alpha_lr_mult_map = normalized_map
                style_cfg_ref['mix_alpha_lr_mult_map'] = mix_alpha_lr_mult_map

    if mix_alpha_lr_mult is not None or mix_alpha_lr_mult_map:
        optimizer_cfg = cfg.get('optimizer', None)
        if optimizer_cfg is None:
            raise ValueError('Optimizer config must exist when mix_alpha_lr_mult is set')
        paramwise_cfg = optimizer_cfg.setdefault('paramwise_cfg', {})
        custom_keys = paramwise_cfg.setdefault('custom_keys', {})

        def _update_custom_key(key: str, lr_mult_value: float):
            entry = dict(custom_keys.get(key, {}))
            entry['lr_mult'] = lr_mult_value
            entry.setdefault('decay_mult', 0.0)
            custom_keys[key] = entry

        if mix_alpha_lr_mult is not None:
            _update_custom_key('backbone.style_mix_alpha', mix_alpha_lr_mult)
            _update_custom_key('backbone.my_module_style_mix_alpha', mix_alpha_lr_mult)

        if mix_alpha_lr_mult_map:
            for stage_key, lr_value in mix_alpha_lr_mult_map.items():
                sanitized = stage_key.replace('.', '_')
                param_key = f'backbone.my_module_style_mix_alpha_{sanitized}'
                _update_custom_key(param_key, lr_value)
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        # re-set gpu_ids with distributed training mode
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    # ensure log_level exists in cfg, provide default if missing
    log_level = cfg.get('log_level', 'INFO')
    # initialize logger with specified or default log level
    logger = get_root_logger(log_file=log_file, log_level=log_level)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info
    meta['config'] = cfg.pretty_text
    # log some basic info
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    # set random seeds
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, '
                    f'deterministic: {args.deterministic}')
        set_random_seed(args.seed, deterministic=args.deterministic)
    cfg.seed = args.seed
    meta['seed'] = args.seed
    meta['exp_name'] = osp.basename(args.config)

    model = build_detector(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))

    # --------freeze & calculate--------
    model.init_weights()

    prototype_manager = None
    prototype_cfg = copy.deepcopy(cfg.get('prototype_adapt', None))
    if args.prototype_file is not None:
        prototype_cfg = prototype_cfg or {}
        prototype_cfg['file'] = args.prototype_file
    if prototype_cfg is not None:
        if 'file' not in prototype_cfg or prototype_cfg['file'] is None:
            raise ValueError('prototype_adapt.file must be provided either in config or via --prototype-file')
        if args.prototype_align_weight is not None:
            prototype_cfg['align_weight'] = args.prototype_align_weight
        if args.prototype_adv_weight is not None:
            prototype_cfg['adv_weight'] = args.prototype_adv_weight
        if args.prototype_temperature is not None:
            prototype_cfg['sample_temperature'] = args.prototype_temperature
        if args.prototype_momentum is not None:
            prototype_cfg['momentum'] = args.prototype_momentum

        align_loss_cfg = prototype_cfg.get('alignment_loss')
        adv_loss_cfg = prototype_cfg.get('adversarial_loss')
        align_weight = prototype_cfg.get('align_weight', 1.0)
        adv_weight = prototype_cfg.get('adv_weight', 1.0)
        sample_temperature = prototype_cfg.get('sample_temperature', -1.0)
        normalize_for_matching = prototype_cfg.get('normalize_for_matching', True)
        momentum = prototype_cfg.get('momentum', 0.0)
        detach_prototypes = prototype_cfg.get('detach_prototypes', True)
        log_interval = prototype_cfg.get('log_interval', 50)
        fallback_on_unlabeled = prototype_cfg.get('fallback_on_unlabeled', True)
        prefer_gt_boxes = prototype_cfg.get('prefer_gt_boxes', True)

        prototype_manager = PrototypeAdvManagerPrototype1001(
            prototype_file=prototype_cfg['file'],
            align_loss_cfg=align_loss_cfg,
            adv_loss_cfg=adv_loss_cfg,
            align_weight=align_weight,
            adv_weight=adv_weight,
            sample_temperature=sample_temperature,
            normalize_for_matching=normalize_for_matching,
            momentum=momentum,
            detach_prototypes=detach_prototypes,
            log_interval=log_interval,
            fallback_on_unlabeled=fallback_on_unlabeled,
            prefer_gt_boxes=prefer_gt_boxes,
        )

        if torch.cuda.is_available():
            current_device = torch.cuda.current_device()
            prototype_manager.to(torch.device('cuda', current_device))
        else:
            prototype_manager.to(torch.device('cpu'))

        if isinstance(prototype_manager.adv_loss, torch.nn.Module):
            registered_children = dict(model.named_children())
            if 'prototype_adv_loss_module' not in registered_children:
                model.add_module('prototype_adv_loss_module', prototype_manager.adv_loss)

        model.prototype_manager = prototype_manager
        if hasattr(model, 'roi_head') and model.roi_head is not None:
            model.roi_head.prototype_manager = prototype_manager
        if hasattr(model, 'bbox_head') and model.bbox_head is not None:
            model.bbox_head.prototype_manager = prototype_manager
        if hasattr(model, 'module') and model.module is not None:
            if hasattr(model.module, 'roi_head') and model.module.roi_head is not None:
                model.module.roi_head.prototype_manager = prototype_manager
            if hasattr(model.module, 'bbox_head') and model.module.bbox_head is not None:
                model.module.bbox_head.prototype_manager = prototype_manager

        cfg.prototype_adapt = prototype_cfg
        logger.info('Prototype manager initialized with file: %s', prototype_cfg['file'])
        logger.info('  align_weight=%.4f, adv_weight=%.4f, momentum=%.4f, temperature=%.4f, fallback_unlabeled=%s, prefer_gt_boxes=%s',
                    align_weight, adv_weight, momentum, sample_temperature, fallback_on_unlabeled, prefer_gt_boxes)
        meta.setdefault('prototype_adapt', prototype_cfg)

    global_proto_manager = None
    global_proto_cfg = copy.deepcopy(cfg.get('global_prototype_adapt', None))
    if args.global_prototype_file is not None:
        global_proto_cfg = global_proto_cfg or {}
        global_proto_cfg['file'] = args.global_prototype_file
    if global_proto_cfg is not None:
        if 'file' not in global_proto_cfg or global_proto_cfg['file'] is None:
            raise ValueError('global_prototype_adapt.file must be provided either in config or via --global-prototype-file')
        if args.global_proto_align_weight is not None:
            global_proto_cfg['align_weight'] = args.global_proto_align_weight
        if args.global_proto_adv_weight is not None:
            global_proto_cfg['adv_weight'] = args.global_proto_adv_weight
        if args.global_proto_feature_source is not None:
            global_proto_cfg['feature_source'] = args.global_proto_feature_source
        if args.global_proto_pool_type is not None:
            global_proto_cfg['pool_type'] = args.global_proto_pool_type

        align_loss_cfg = global_proto_cfg.get('alignment_loss')
        adv_loss_cfg = global_proto_cfg.get('adversarial_loss')
        align_weight = global_proto_cfg.get('align_weight', 1.0)
        adv_weight = global_proto_cfg.get('adv_weight', 1.0)
        normalize_for_matching = global_proto_cfg.get('normalize_for_matching', True)
        detach_prototypes = global_proto_cfg.get('detach_prototypes', True)
        feature_source = global_proto_cfg.get('feature_source', 'c4')
        pool_type = global_proto_cfg.get('pool_type', 'avg')
        log_interval = global_proto_cfg.get('log_interval', 50)

        global_proto_manager = GlobalPrototypeAdvManager(
            prototype_file=global_proto_cfg['file'],
            align_loss_cfg=align_loss_cfg,
            adv_loss_cfg=adv_loss_cfg,
            align_weight=align_weight,
            adv_weight=adv_weight,
            normalize_for_matching=normalize_for_matching,
            detach_prototypes=detach_prototypes,
            feature_source=feature_source,
            pool_type=pool_type,
            log_interval=log_interval,
        )

        if torch.cuda.is_available():
            current_device = torch.cuda.current_device()
            global_proto_manager.to(torch.device('cuda', current_device))
        else:
            global_proto_manager.to(torch.device('cpu'))

        if isinstance(global_proto_manager.adv_loss, torch.nn.Module):
            registered_children = dict(model.named_children())
            if 'global_proto_adv_loss_module' not in registered_children:
                model.add_module('global_proto_adv_loss_module', global_proto_manager.adv_loss)

        model.global_proto_manager = global_proto_manager
        if hasattr(model, 'module') and model.module is not None:
            model.module.global_proto_manager = global_proto_manager

        cfg.global_prototype_adapt = global_proto_cfg
        logger.info('Global prototype manager initialized with file: %s', global_proto_cfg['file'])
        logger.info('  feature_source=%s, pool_type=%s, align_weight=%.4f, adv_weight=%.4f, detach_prototypes=%s',
                    feature_source, pool_type, align_weight, adv_weight, detach_prototypes)
        meta.setdefault('global_prototype_adapt', global_proto_cfg)

    frequency_proto_manager = None
    frequency_proto_cfg = copy.deepcopy(cfg.get('frequency_prototype_adapt', None))
    if args.frequency_prototype_file is not None:
        frequency_proto_cfg = frequency_proto_cfg or {}
        frequency_proto_cfg['file'] = args.frequency_prototype_file
    if frequency_proto_cfg is not None:
        if 'file' not in frequency_proto_cfg or frequency_proto_cfg['file'] is None:
            raise ValueError('frequency_prototype_adapt.file must be provided either in config or via --frequency-prototype-file')
        if args.frequency_proto_align_weight is not None:
            frequency_proto_cfg['align_weight'] = args.frequency_proto_align_weight
        if args.frequency_proto_adv_weight is not None:
            frequency_proto_cfg['adv_weight'] = args.frequency_proto_adv_weight
        if args.frequency_proto_pool_size is not None:
            frequency_proto_cfg['freq_pool_size'] = args.frequency_proto_pool_size
        if args.frequency_proto_log_amplitude != 'auto':
            frequency_proto_cfg['fft_log_amplitude'] = bool(args.frequency_proto_log_amplitude == 'on')
        if args.frequency_proto_shift != 'auto':
            frequency_proto_cfg['fft_shift'] = bool(args.frequency_proto_shift == 'on')
        if args.frequency_proto_channel_aggregation != 'auto':
            frequency_proto_cfg['channel_aggregation'] = args.frequency_proto_channel_aggregation
        if args.frequency_proto_input_source != 'auto':
            frequency_proto_cfg['input_source'] = args.frequency_proto_input_source
        if args.frequency_proto_feature_stage is not None:
            frequency_proto_cfg['feature_stage'] = args.frequency_proto_feature_stage
        if args.frequency_proto_channel_subsample is not None:
            frequency_proto_cfg['channel_subsample'] = args.frequency_proto_channel_subsample

        align_loss_cfg = frequency_proto_cfg.get('alignment_loss')
        adv_loss_cfg = frequency_proto_cfg.get('adversarial_loss')
        align_weight = frequency_proto_cfg.get('align_weight', 1.0)
        adv_weight = frequency_proto_cfg.get('adv_weight', 1.0)
        normalize_for_matching = frequency_proto_cfg.get('normalize_for_matching', True)
        detach_prototypes = frequency_proto_cfg.get('detach_prototypes', True)
        freq_pool_size = frequency_proto_cfg.get('freq_pool_size', 48)
        fft_log_amplitude = frequency_proto_cfg.get('fft_log_amplitude', True)
        fft_shift = frequency_proto_cfg.get('fft_shift', True)
        channel_aggregation = frequency_proto_cfg.get('channel_aggregation', 'stack')
        channel_subsample = frequency_proto_cfg.get('channel_subsample', None)
        input_source = frequency_proto_cfg.get('input_source', 'image')
        feature_stage = frequency_proto_cfg.get('feature_stage', 'c5')
        log_interval = frequency_proto_cfg.get('log_interval', 50)

        frequency_proto_manager = FrequencyPrototypeAdvManager(
            prototype_file=frequency_proto_cfg['file'],
            align_loss_cfg=align_loss_cfg,
            adv_loss_cfg=adv_loss_cfg,
            align_weight=align_weight,
            adv_weight=adv_weight,
            normalize_for_matching=normalize_for_matching,
            detach_prototypes=detach_prototypes,
            log_interval=log_interval,
            freq_pool_size=freq_pool_size,
            fft_log_amplitude=fft_log_amplitude,
            fft_shift=fft_shift,
            channel_aggregation=channel_aggregation,
            channel_subsample=channel_subsample,
            input_source=input_source,
            feature_stage=feature_stage,
        )

        if torch.cuda.is_available():
            current_device = torch.cuda.current_device()
            frequency_proto_manager.to(torch.device('cuda', current_device))
        else:
            frequency_proto_manager.to(torch.device('cpu'))

        if isinstance(frequency_proto_manager.adv_loss, torch.nn.Module):
            registered_children = dict(model.named_children())
            if 'frequency_proto_adv_loss_module' not in registered_children:
                model.add_module('frequency_proto_adv_loss_module', frequency_proto_manager.adv_loss)

        model.frequency_proto_manager = frequency_proto_manager
        if hasattr(model, 'module') and model.module is not None:
            model.module.frequency_proto_manager = frequency_proto_manager

        cfg.frequency_prototype_adapt = frequency_proto_cfg
        logger.info('Frequency prototype manager initialized with file: %s', frequency_proto_cfg['file'])
        logger.info('  source=%s, stage=%s, pool_size=%s, fft_log=%s, fft_shift=%s, channel_agg=%s, align_weight=%.4f, adv_weight=%.4f, detach_prototypes=%s',
                    input_source, feature_stage, freq_pool_size, fft_log_amplitude, fft_shift, channel_aggregation,
                    align_weight, adv_weight, detach_prototypes)
        meta.setdefault('frequency_prototype_adapt', frequency_proto_cfg)

    if distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        if prototype_manager is not None:
            if hasattr(model, 'roi_head') and model.roi_head is not None:
                model.roi_head.prototype_manager = prototype_manager
            if hasattr(model, 'bbox_head') and model.bbox_head is not None:
                model.bbox_head.prototype_manager = prototype_manager
        if global_proto_manager is not None:
            model.global_proto_manager = global_proto_manager
        if frequency_proto_manager is not None:
            model.frequency_proto_manager = frequency_proto_manager

    my_module_num = 0
    backbone_num = 0
    trained_backbone_num = 0
    # 新增：统计对抗判别器参数量
    adv_num = 0
    adv_trainable_num = 0
    # 新增：用于存储详细参数信息的列表
    param_details = []
    architecture_lines = summarize_model_architecture(model)
    if architecture_lines:
        logger.info("Model architecture overview:\n%s", "\n".join(architecture_lines))
    logger.info(model)
    for name, param in model.named_parameters():
        param_count = param.numel() 

        # 为日志摘要计算统计信息（保持不变）
        if 'backbone' in name:
            backbone_num += param_count
            if param.requires_grad:
                trained_backbone_num += param_count
        
        if 'ladder' in name or 'my_module' in name or 'adapter' in name:
            my_module_num += param_count

        # 统计判别器（adv_loss）参数
        if 'prototype_adv_loss_module' in name:
            adv_num += param_count
            if param.requires_grad:
                adv_trainable_num += param_count

        # 仅将可训练的参数信息添加到列表中，用于生成JSON文件
        if param.requires_grad:
            param_details.append({
                "name": name,
                "count": param_count
            })


    logger.info("=" * 80)
    logger.info("MODEL PARAMETER STATISTICS")
    logger.info("=" * 80)
    logger.info(f"Total backbone parameters: {backbone_num:,}")
    logger.info(f"Trainable backbone parameters: {trained_backbone_num:,}")
    logger.info(f"Custom module parameters (ladder/my_module/adapter): {my_module_num:,}")
    logger.info("-" * 80)
    if backbone_num > 0:
        custom_vs_backbone = my_module_num / backbone_num * 100
        logger.info(f"Custom modules vs total backbone: {custom_vs_backbone:.2f}%")
    trainable_ratio = trained_backbone_num / backbone_num * 100 if backbone_num > 0 else 0
    logger.info(f"Trainable backbone parameters ratio: {trainable_ratio:.2f}%")
    # 输出判别器参数统计
    logger.info(f"Total adversarial discriminator parameters: {adv_num:,}")
    logger.info(f"Trainable adversarial discriminator parameters: {adv_trainable_num:,}")
    logger.info("=" * 80)
    
    # 新增：将详细参数信息保存为 JSON 文件
    try:
        json_output_path = osp.join(cfg.work_dir, 'parameter_statistics.json')
        with open(json_output_path, 'w') as f:
            json.dump(param_details, f, indent=4)
        logger.info(f"Detailed trainable parameter statistics saved to {json_output_path}")
    except Exception as e:
        logger.error(f"Failed to save parameter statistics JSON file: {e}")
    logger.info("=" * 80)
    # --------freeze & calculate--------

    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.pipeline
        datasets.append(build_dataset(val_dataset))
    if cfg.checkpoint_config is not None:
        # save mmdet version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmdet_version=__version__ + get_git_hash()[:7],
            CLASSES=datasets[0].CLASSES)
    # add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES
    train_detector(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=(not args.no_validate),
        timestamp=timestamp,
        meta=meta)


if __name__ == '__main__':
    main()

