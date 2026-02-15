import copy
import os
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from mmdet.models.builder import build_loss


class GlobalPrototypeAdvManager:
    """Manage image-level prototype alignment and adversarial losses."""

    def __init__(
        self,
        prototype_file: str,
        align_loss_cfg: Optional[Dict] = None,
        adv_loss_cfg: Optional[Dict] = None,
        align_weight: float = 1.0,
        adv_weight: float = 1.0,
        normalize_for_matching: bool = True,
        detach_prototypes: bool = True,
        feature_source: str = 'c4',
        pool_type: str = 'avg',
        log_interval: int = 50,
    ) -> None:
        if not os.path.isfile(prototype_file):
            raise FileNotFoundError(f'Global prototype file not found: {prototype_file}')
        payload = torch.load(prototype_file, map_location='cpu')
        if not isinstance(payload, dict) or 'global_proto' not in payload:
            raise ValueError('Global prototype file must contain a "global_proto" key.')

        global_proto = payload['global_proto']
        if not isinstance(global_proto, dict) or 'centers' not in global_proto:
            raise ValueError('Global prototype dictionary must contain "centers" tensor.')

        centers = global_proto['centers']
        if not isinstance(centers, torch.Tensor):
            raise TypeError('Global prototype centers must be a torch.Tensor')
        if centers.dim() != 2:
            raise ValueError('Global prototype centers must be a 2D tensor [num_proto, dim].')

        self.prototype_file = prototype_file
        self.metadata = payload.get('metadata', {})
        self.device = torch.device('cpu')
        self.feature_dim: int = int(centers.size(1))

        self.prototypes = centers.clone()
        self.prototype_counts = global_proto.get('counts')
        self.align_loss_cfg = copy.deepcopy(align_loss_cfg) if align_loss_cfg else None
        self.adv_loss_cfg = copy.deepcopy(adv_loss_cfg) if adv_loss_cfg else None
        self.align_loss = build_loss(self.align_loss_cfg) if self.align_loss_cfg else None
        self.adv_loss = build_loss(self._prepare_adv_cfg(self.adv_loss_cfg)) if self.adv_loss_cfg else None

        self.align_weight = align_weight
        self.adv_weight = adv_weight
        self.normalize_for_matching = normalize_for_matching
        self.detach_prototypes = detach_prototypes
        self.feature_source = feature_source
        self.pool_type = pool_type
        self.log_interval = log_interval
        self._iter = 0

    def _prepare_adv_cfg(self, cfg: Optional[Dict]) -> Optional[Dict]:
        if cfg is None:
            return None
        cfg = copy.deepcopy(cfg)
        cfg.setdefault('in_dim', self.feature_dim)
        return cfg

    def to(self, device: torch.device) -> 'GlobalPrototypeAdvManager':
        self.device = device
        self.prototypes = self.prototypes.to(device)
        if self.prototype_counts is not None and isinstance(self.prototype_counts, torch.Tensor):
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

    # ------------------------------------------------------------------
    # Feature selection utilities
    # ------------------------------------------------------------------
    def _unwrap_feature(self, feat):
        if isinstance(feat, (list, tuple)):
            return feat[-1]
        if isinstance(feat, dict):
            if not feat:
                return None
            last_key = list(feat.keys())[-1]
            return feat[last_key]
        return feat

    def _select_feature_map(self, backbone_feats, neck_feats):
        if self.feature_source == 'c4':
            target = backbone_feats
        elif self.feature_source == 'p5':
            target = neck_feats
        else:
            raise ValueError(f'Unsupported feature_source: {self.feature_source}')
        return self._unwrap_feature(target)

    def _global_pool(self, feature_map: torch.Tensor) -> torch.Tensor:
        if feature_map is None:
            return None
        if feature_map.dim() != 4:
            raise ValueError(f'Expected feature map with shape (N, C, H, W), got {tuple(feature_map.shape)}')
        if self.pool_type == 'avg':
            pooled = feature_map.mean(dim=(2, 3))
        elif self.pool_type == 'avgmax':
            pooled_avg = feature_map.mean(dim=(2, 3))
            pooled_max = feature_map.amax(dim=(2, 3))
            pooled = torch.cat([pooled_avg, pooled_max], dim=1)
        else:
            raise ValueError(f'Unsupported pool_type: {self.pool_type}')
        return pooled

    def collect_global_tokens(self, backbone_feats, neck_feats) -> Optional[torch.Tensor]:
        feature_map = self._select_feature_map(backbone_feats, neck_feats)
        if feature_map is None:
            return None
        pooled = self._global_pool(feature_map)
        return pooled

    # ------------------------------------------------------------------
    # Loss computation
    # ------------------------------------------------------------------
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

    def compute_losses(self, features: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
        self._iter += 1
        if features is None or features.numel() == 0:
            return {}, {}
        features = features.to(self.device)
        valid_mask = torch.ones(features.size(0), dtype=torch.bool, device=features.device)
        matched, scores = self._match_prototypes(features)

        losses: Dict[str, torch.Tensor] = {}
        stats: Dict[str, float] = {}

        if self.align_loss is not None and valid_mask.any():
            align_val = self.align_loss(features, matched, valid_mask)
            losses['loss_global_proto_align'] = align_val * self.align_weight
            stats['global_proto_align'] = float(align_val.detach().cpu().item())

        if self.adv_loss is not None and valid_mask.any():
            proto_feats = matched if matched is not None else None
            if proto_feats is not None and self.detach_prototypes:
                proto_feats = proto_feats.detach()
            adv_val = self.adv_loss(features, proto_feats)
            losses['loss_global_proto_adv'] = adv_val * self.adv_weight
            stats['global_proto_adv_bce'] = float(adv_val.detach().cpu().item())

        if scores is not None and scores.numel() > 0:
            stats['global_proto_match_score'] = float(scores.mean().detach().cpu().item())

        return losses, stats

    def compute_from_model_outputs(self, backbone_feats, neck_feats) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
        tokens = self.collect_global_tokens(backbone_feats, neck_feats)
        if tokens is None:
            return {}, {}
        return self.compute_losses(tokens)
