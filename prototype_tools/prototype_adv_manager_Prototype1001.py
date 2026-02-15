import copy
import os
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from mmdet.models.builder import build_loss

from .prototype_bank_Prototype1001 import PrototypeBankPrototype1001


class PrototypeAdvManagerPrototype1001:
    """Manage prototype-driven alignment and adversarial losses during training."""

    def __init__(
        self,
        prototype_file: str,
        align_loss_cfg: Optional[Dict] = None,
        adv_loss_cfg: Optional[Dict] = None,
        align_weight: float = 1.0,
        adv_weight: float = 1.0,
        sample_temperature: float = -1.0,
        normalize_for_matching: bool = True,
        momentum: float = 0.0,   # 决定是否使用动量更新原型
        detach_prototypes: bool = True,
        log_interval: int = 50,
        fallback_on_unlabeled: bool = True,
        prefer_gt_boxes: bool = True,
    ) -> None:
        if not os.path.isfile(prototype_file):
            raise FileNotFoundError(f'Prototype file not found: {prototype_file}')
        payload = torch.load(prototype_file, map_location='cpu')
        if not isinstance(payload, dict) or 'prototypes' not in payload:
            raise ValueError('Prototype file must contain a "prototypes" dictionary.')
        self.prototype_file = prototype_file
        self.metadata = payload.get('metadata', {})
        self.device = torch.device('cpu')
        self.bank = PrototypeBankPrototype1001(payload['prototypes'], self.device, momentum=momentum)

        self.feature_dim: Optional[int] = self.bank.get_feature_dim()
        self.align_loss_cfg = copy.deepcopy(align_loss_cfg) if align_loss_cfg else None
        self.adv_loss_cfg = copy.deepcopy(adv_loss_cfg) if adv_loss_cfg else None
        self.align_loss = build_loss(self.align_loss_cfg) if self.align_loss_cfg else None
        prepared_adv_cfg = self._prepare_adv_cfg(self.adv_loss_cfg, self.feature_dim)
        self.adv_loss_cfg_resolved = prepared_adv_cfg
        self.adv_loss = build_loss(prepared_adv_cfg) if prepared_adv_cfg else None

        self.align_weight = align_weight
        self.adv_weight = adv_weight
        self.sample_temperature = sample_temperature
        self.normalize_for_matching = normalize_for_matching
        self.detach_prototypes = detach_prototypes
        self.log_interval = log_interval
        self._iter = 0
        self.fallback_on_unlabeled = fallback_on_unlabeled
        self.prefer_gt_boxes = prefer_gt_boxes

    def _prepare_adv_cfg(self, cfg: Optional[Dict], feature_dim: Optional[int] = None) -> Optional[Dict]:
        if cfg is None:
            return None
        cfg = copy.deepcopy(cfg)
        if feature_dim is None:
            feature_dim = self.feature_dim
        if feature_dim is not None:
            cfg.setdefault('in_dim', feature_dim)
        return cfg

    def to(self, device: torch.device) -> 'PrototypeAdvManagerPrototype1001':
        self.device = device
        self.bank.to(device)
        if self.align_loss is not None and hasattr(self.align_loss, 'to'):
            self.align_loss = self.align_loss.to(device)
        if self.adv_loss is not None and hasattr(self.adv_loss, 'to'):
            self.adv_loss = self.adv_loss.to(device)
        return self

    def _ensure_adv_loss(self, feature_dim: int) -> None:
        if self.adv_loss is not None or self.adv_loss_cfg is None:
            return
        cfg = self._prepare_adv_cfg(self.adv_loss_cfg, feature_dim)
        self.adv_loss_cfg_resolved = cfg
        if cfg is None or cfg.get('in_dim') is None:
            return
        self.adv_loss = build_loss(cfg)
        if hasattr(self.adv_loss, 'to'):
            self.adv_loss = self.adv_loss.to(self.device)

    def _match_prototypes(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        original_labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if features.numel() == 0:
            zero_proto = features.new_zeros(features.size(0), features.size(1))
            empty_bool = features.new_zeros(features.size(0), dtype=torch.bool)
            empty_long = features.new_full((features.size(0),), -1, dtype=torch.long)
            empty_labels = features.new_full((features.size(0),), -1, dtype=torch.long)
            return zero_proto, empty_bool, empty_long, empty_bool.float(), empty_labels

        matched = features.new_zeros(features.size())
        valid_mask = torch.zeros(features.size(0), dtype=torch.bool, device=features.device)
        proto_indices = torch.full((features.size(0),), -1, dtype=torch.long, device=features.device)
        cos_scores = torch.zeros(features.size(0), dtype=torch.float32, device=features.device)
        assigned_labels = labels.clone()

        for idx, label_tensor in enumerate(labels):
            cls_id = int(label_tensor.item())
            orig_cls_id = int(original_labels[idx].item()) if original_labels is not None else cls_id
            if cls_id in self.bank.centers:
                centers = self.bank.get_all(cls_id)
                if centers.numel() == 0:
                    continue

                if self.sample_temperature is not None and self.sample_temperature > 0:
                    center, center_idx = self.bank.sample_center(cls_id, temperature=self.sample_temperature)
                else:
                    if self.normalize_for_matching:
                        feat_norm = F.normalize(features[idx:idx + 1], dim=1)
                        centers_norm = F.normalize(centers, dim=1)
                        sims = torch.mm(feat_norm, centers_norm.t()).squeeze(0)
                        center_idx = int(torch.argmax(sims).item())
                        center = centers[center_idx]
                        cos_scores[idx] = sims[center_idx]
                    else:
                        dists = torch.cdist(features[idx:idx + 1], centers, p=2.0).squeeze(0)
                        center_idx = int(torch.argmin(dists).item())
                        center = centers[center_idx]
                        cos_scores[idx] = 1.0 - dists[center_idx]

                matched[idx] = center.to(features.device)
                valid_mask[idx] = True
                proto_indices[idx] = center_idx
                if cos_scores[idx] == 0 and self.normalize_for_matching:
                    feat_norm = F.normalize(features[idx:idx + 1], dim=1)
                    center_norm = F.normalize(center.unsqueeze(0), dim=1)
                    cos_scores[idx] = torch.mm(feat_norm, center_norm.t()).squeeze(0)
                continue

            if not self.fallback_on_unlabeled or orig_cls_id >= 0:
                continue

            best_cls, center_idx, center, score = self.bank.find_best_match(features[idx], self.normalize_for_matching)
            if best_cls is None or center is None:
                continue
            matched[idx] = center.to(features.device)
            valid_mask[idx] = True
            proto_indices[idx] = center_idx if center_idx is not None else -1
            cos_scores[idx] = score if score is not None else 0.0
            assigned_labels[idx] = int(best_cls)

        return matched, valid_mask, proto_indices, cos_scores, assigned_labels

    def set_gradient_reversal_weight(self, weight: float) -> None:
        if self.adv_loss is not None and hasattr(self.adv_loss, 'gradient_reversal'):
            self.adv_loss.gradient_reversal.lambd = float(weight)
        if self.adv_loss_cfg_resolved is not None:
            self.adv_loss_cfg_resolved['gradient_reverse_weight'] = float(weight)

    def compute_losses(
        self, features: torch.Tensor, labels: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
        self._iter += 1
        if features.size(0) == 0:
            return {}, {}
        if self.feature_dim is None and features.size(1) > 0:
            self.feature_dim = int(features.size(1))
        features = features.to(self.device)
        labels = labels.to(self.device)
        labels_for_matching = labels

        matched, valid_mask, _, cos_scores, assigned_labels = self._match_prototypes(
            features, labels_for_matching, original_labels=labels)

        losses: Dict[str, torch.Tensor] = {}
        stats: Dict[str, float] = {}

        if self.align_loss is not None and valid_mask.any():
            align_val = self.align_loss(features, matched, valid_mask)
            losses['loss_proto_align'] = align_val * self.align_weight

        if self.adv_loss_cfg is not None:
            self._ensure_adv_loss(features.size(1))
            if self.adv_loss is not None and valid_mask.any():
                target_feats = features[valid_mask]
                proto_feats = matched[valid_mask] if matched is not None else None
                if self.detach_prototypes and proto_feats is not None:
                    proto_feats = proto_feats.detach()
                adv_val = self.adv_loss(target_feats, proto_feats)
                losses['loss_proto_adv'] = adv_val * self.adv_weight
                stats['prototype_adv_bce'] = float(adv_val.item())

        if getattr(self.bank, 'momentum', 0.0) > 0.0 and valid_mask.any():
            effective_labels = assigned_labels
            for cls_id in effective_labels[valid_mask].unique():
                if cls_id.item() < 0:
                    continue
                cls_mask = (effective_labels == cls_id) & valid_mask
                if cls_mask.any():
                    self.bank.update(int(cls_id.item()), features[cls_mask])

        if valid_mask.any():
            stats['prototype_cos_mean'] = float(cos_scores[valid_mask].mean().item())
            stats['prototype_match_ratio'] = float(valid_mask.float().mean().item())
            stats['prototype_matches'] = int(valid_mask.sum().item())
            fallback_mask = valid_mask & (assigned_labels >= 0) & (labels_for_matching < 0)
            if fallback_mask.any():
                stats['prototype_fallback_matches'] = int(fallback_mask.sum().item())
        else:
            stats['prototype_cos_mean'] = 0.0
            stats['prototype_match_ratio'] = 0.0
            stats['prototype_matches'] = 0
        stats['prototype_total'] = int(valid_mask.numel())

        return losses, stats
