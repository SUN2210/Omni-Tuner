import copy
import os
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from mmdet.models.builder import build_loss


class FrequencyPrototypeAdvManager:
    """管理频域图级原型的对齐与对抗损失。"""

    def __init__(
        self,
        prototype_file: str,
        align_loss_cfg: Optional[Dict] = None,
        adv_loss_cfg: Optional[Dict] = None,
        align_weight: float = 1.0,
        adv_weight: float = 1.0,
        normalize_for_matching: bool = True,
        detach_prototypes: bool = True,
        log_interval: int = 50,
        freq_pool_size: int = 48,
        fft_log_amplitude: bool = True,
        fft_shift: bool = True,
        channel_aggregation: str = 'stack',
        channel_subsample: Optional[int] = None,
        input_source: str = 'image',
        feature_stage: str = 'c5',
    ) -> None:
        if not os.path.isfile(prototype_file):
            raise FileNotFoundError(f'Frequency prototype file not found: {prototype_file}')
        payload = torch.load(prototype_file, map_location='cpu')
        if not isinstance(payload, dict) or 'freq_proto' not in payload:
            raise ValueError('Frequency prototype file must contain a "freq_proto" dictionary.')
        freq_proto = payload['freq_proto']
        if not isinstance(freq_proto, dict) or 'centers' not in freq_proto:
            raise ValueError('freq_proto dictionary must contain "centers" tensor.')

        centers = freq_proto['centers']
        if not isinstance(centers, torch.Tensor):
            raise TypeError('Frequency prototype centers must be a torch.Tensor')
        if centers.dim() != 2:
            raise ValueError('Frequency prototype centers must be 2D [num_proto, dim].')

        self.prototype_file = prototype_file
        self.metadata = payload.get('metadata', {})
        self.device = torch.device('cpu')
        self.prototypes = centers.clone()
        self.prototype_counts = freq_proto.get('counts')
        self.feature_dim: int = int(centers.size(1))

        self.align_loss_cfg = copy.deepcopy(align_loss_cfg) if align_loss_cfg else None
        self.adv_loss_cfg = copy.deepcopy(adv_loss_cfg) if adv_loss_cfg else None
        self.align_loss = build_loss(self.align_loss_cfg) if self.align_loss_cfg else None
        self.adv_loss = build_loss(self._prepare_adv_cfg(self.adv_loss_cfg)) if self.adv_loss_cfg else None

        self.align_weight = align_weight
        self.adv_weight = adv_weight
        self.normalize_for_matching = normalize_for_matching
        self.detach_prototypes = detach_prototypes
        self.log_interval = log_interval
        self._iter = 0

        self.freq_pool_size = freq_pool_size
        self.fft_log_amplitude = fft_log_amplitude
        self.fft_shift = fft_shift
        self.channel_aggregation = channel_aggregation
        self.channel_subsample = channel_subsample
        self.input_source = input_source
        self.feature_stage = feature_stage

    def _prepare_adv_cfg(self, cfg: Optional[Dict]) -> Optional[Dict]:
        if cfg is None:
            return None
        cfg = copy.deepcopy(cfg)
        cfg.setdefault('in_dim', self.feature_dim)
        return cfg

    def to(self, device: torch.device) -> 'FrequencyPrototypeAdvManager':
        self.device = device
        self.prototypes = self.prototypes.to(device)
        if isinstance(self.prototype_counts, torch.Tensor):
            self.prototype_counts = self.prototype_counts.to(device)
        if self.align_loss is not None and hasattr(self.align_loss, 'to'):
            self.align_loss = self.align_loss.to(device)
        if self.adv_loss is not None and hasattr(self.adv_loss, 'to'):
            self.adv_loss = self.adv_loss.to(device)
        return self

    def set_gradient_reversal_weight(self, weight: float) -> None:
        if self.adv_loss is not None and hasattr(self.adv_loss, 'gradient_reversal'):
            self.adv_loss.gradient_reversal.lambd = float(weight)
        if self.adv_loss_cfg is not None:
            self.adv_loss_cfg.setdefault('gradient_reverse_weight', float(weight))

    def _match_prototypes(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if features.numel() == 0:
            empty_proto = features.new_zeros(features.size(0), self.feature_dim)
            empty_scores = features.new_zeros(features.size(0))
            return empty_proto, empty_scores
        prototypes = self.prototypes.to(features.device)
        if self.normalize_for_matching:
            feats_norm = F.normalize(features, dim=1)
            protos_norm = F.normalize(prototypes, dim=1)
            sims = torch.matmul(feats_norm, protos_norm.t())
            indices = torch.argmax(sims, dim=1)
            scores = sims.gather(1, indices.unsqueeze(1)).squeeze(1)
        else:
            dists = torch.cdist(features, prototypes)
            indices = torch.argmin(dists, dim=1)
            scores = -dists.gather(1, indices.unsqueeze(1)).squeeze(1)
        matched = prototypes.index_select(0, indices)
        return matched, scores

    def _fft_to_tokens(self, tensor: torch.Tensor) -> torch.Tensor:
        freq_map = torch.fft.fft2(tensor, dim=(-2, -1))
        magnitude = torch.abs(freq_map)
        if self.fft_shift:
            magnitude = torch.fft.fftshift(magnitude, dim=(-2, -1))
        if self.fft_log_amplitude:
            magnitude = torch.log1p(magnitude)

        if self.channel_aggregation == 'mean':
            magnitude = magnitude.mean(dim=1, keepdim=True)
        elif self.channel_aggregation == 'max':
            magnitude = magnitude.amax(dim=1, keepdim=True)
        elif self.channel_aggregation == 'stack':
            if self.channel_subsample is not None:
                magnitude = magnitude[:, :self.channel_subsample]
            pass
        else:
            raise ValueError(f'Unsupported channel aggregation: {self.channel_aggregation}')

        if self.freq_pool_size > 0:
            magnitude = F.adaptive_avg_pool2d(magnitude, (self.freq_pool_size, self.freq_pool_size))

        tokens = magnitude.reshape(magnitude.size(0), -1)
        return tokens

    def _select_backbone_stage(self, backbone_out) -> torch.Tensor:
        if backbone_out is None:
            raise ValueError('backbone outputs are required when input_source is "backbone".')
        if isinstance(backbone_out, (list, tuple)):
            mapping = {'c2': 0, 'c3': 1, 'c4': 2, 'c5': 3}
            if self.feature_stage == 'last':
                idx = len(backbone_out) - 1
            else:
                idx = mapping.get(self.feature_stage, len(backbone_out) - 1)
                idx = min(idx, len(backbone_out) - 1)
            feat = backbone_out[idx]
        elif isinstance(backbone_out, dict):
            keys = list(backbone_out.keys())
            if self.feature_stage == 'last':
                key = keys[-1]
            else:
                mapping = {'c2': 0, 'c3': 1, 'c4': 2, 'c5': 3}
                key = keys[mapping.get(self.feature_stage, len(keys) - 1)]
            feat = backbone_out[key]
        else:
            feat = backbone_out
        if isinstance(feat, (list, tuple)):
            feat = feat[-1]
        if not torch.is_tensor(feat):
            raise TypeError(f'Backbone stage output must be tensor, got {type(feat)}')
        return feat

    def _compute_tokens_from_image(self, imgs: torch.Tensor) -> Optional[torch.Tensor]:
        if imgs is None or not torch.is_tensor(imgs):
            return None
        if imgs.dim() != 4:
            raise ValueError(f'Expected image tensor of shape (N, C, H, W), got {tuple(imgs.shape)}')
        return self._fft_to_tokens(imgs)

    def _compute_tokens_from_backbone(self, backbone_feats) -> Optional[torch.Tensor]:
        stage_feat = self._select_backbone_stage(backbone_feats)
        if stage_feat is None or not torch.is_tensor(stage_feat):
            return None
        return self._fft_to_tokens(stage_feat)

    def compute_losses(self, features: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
        self._iter += 1
        if features is None or features.numel() == 0:
            return {}, {}
        features = features.to(self.device)
        matched, scores = self._match_prototypes(features)
        valid_mask = torch.ones(features.size(0), dtype=torch.bool, device=features.device)

        losses: Dict[str, torch.Tensor] = {}
        stats: Dict[str, float] = {}

        if self.align_loss is not None and valid_mask.any():
            align_val = self.align_loss(features, matched, valid_mask)
            losses['loss_freq_proto_align'] = align_val * self.align_weight
            stats['freq_proto_align'] = float(align_val.detach().cpu().item())

        if self.adv_loss is not None and valid_mask.any():
            proto_feats = matched if matched is not None else None
            if proto_feats is not None and self.detach_prototypes:
                proto_feats = proto_feats.detach()
            adv_val = self.adv_loss(features, proto_feats)
            losses['loss_freq_proto_adv'] = adv_val * self.adv_weight
            stats['freq_proto_adv_bce'] = float(adv_val.detach().cpu().item())

        if scores is not None and scores.numel() > 0:
            stats['freq_proto_match_score'] = float(scores.mean().detach().cpu().item())

        return losses, stats

    def compute_from_model_outputs(
        self,
        imgs: Optional[torch.Tensor] = None,
        backbone_feats=None,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
        if self.input_source == 'image':
            tokens = self._compute_tokens_from_image(imgs)
        elif self.input_source == 'backbone':
            tokens = self._compute_tokens_from_backbone(backbone_feats)
        else:
            raise ValueError(f'Unsupported input_source: {self.input_source}')
        if tokens is None:
            return {}, {}
        return self.compute_losses(tokens)
