import os
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import PROTOTYPE_BANKS


@PROTOTYPE_BANKS.register_module()
class PrototypeBank(nn.Module):
    """Manage class-wise prototypes loaded from an offline extraction script.

    The bank stores per-class prototype centers and optional counts, supports
    retrieving nearest prototypes for given features, and exposes metadata for
    downstream logging. Prototypes are stored as buffers so they follow the
    module's device placement and do not require gradients.
    """

    def __init__(
        self,
        file_path: str,
        normalize: bool = True,
        default_to_identity: bool = False,
    ) -> None:
        super().__init__()
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f'Prototype file not found: {file_path}')
        payload = torch.load(file_path, map_location='cpu')
        if not isinstance(payload, dict) or 'prototypes' not in payload:
            raise ValueError('Prototype file missing "prototypes" dictionary.')

        prototypes: Dict = payload['prototypes']
        metadata: Dict = payload.get('metadata', {})
        class_names = metadata.get('class_names')
        if class_names is not None:
            num_classes = len(class_names)
        else:
            num_classes = max(int(k) for k in prototypes.keys()) + 1 if prototypes else 0

        feature_dim = metadata.get('feature_dim', None)
        num_prototypes = metadata.get('num_prototypes', None)

        if num_prototypes is None:
            num_prototypes = 0
            for entry in prototypes.values():
                centers = entry.get('centers')
                if centers is None:
                    continue
                num_prototypes = max(num_prototypes, len(centers))
        if feature_dim is None:
            for entry in prototypes.values():
                centers = entry.get('centers')
                if centers is None:
                    continue
                centers_tensor = torch.as_tensor(centers)
                feature_dim = centers_tensor.size(-1)
                break
        if feature_dim is None or feature_dim <= 0:
            raise ValueError('Unable to infer prototype feature dimension.')
        if num_prototypes <= 0:
            raise ValueError('Prototype bank requires positive num_prototypes.')

        centers_tensor = torch.zeros(num_classes, num_prototypes, feature_dim, dtype=torch.float32)
        counts_tensor = torch.zeros(num_classes, num_prototypes, dtype=torch.float32)
        valid_mask = torch.zeros(num_classes, num_prototypes, dtype=torch.bool)

        for raw_cls_id, entry in prototypes.items():
            cls_id = int(raw_cls_id)
            if cls_id >= num_classes:
                continue
            centers = entry.get('centers')
            if centers is None:
                continue
            cls_centers = torch.as_tensor(centers, dtype=torch.float32)
            k = min(cls_centers.size(0), num_prototypes)
            centers_tensor[cls_id, :k] = cls_centers[:k]
            valid_mask[cls_id, :k] = True
            counts = entry.get('counts')
            if counts is not None:
                counts_tensor[cls_id, :k] = torch.as_tensor(counts, dtype=torch.float32)[:k]

        self.register_buffer('centers', centers_tensor)
        self.register_buffer('counts', counts_tensor)
        self.register_buffer('valid_mask', valid_mask)

        self.normalize = normalize
        self.default_to_identity = default_to_identity
        self.num_classes = num_classes
        self.num_prototypes = num_prototypes
        self.feature_dim = feature_dim
        self.metadata = metadata
        self.prototype_path = file_path

    def forward(self, *args, **kwargs):  # pragma: no cover - not used
        raise RuntimeError('PrototypeBank is a data container and should not be called directly.')

    def extra_repr(self) -> str:
        return (f'num_classes={self.num_classes}, num_prototypes={self.num_prototypes}, '
                f'feature_dim={self.feature_dim}, normalize={self.normalize}, '
                f'path={self.prototype_path}')

    def get_all(self, cls_id: int) -> torch.Tensor:
        """Return all prototypes for a class as a tensor of shape (K, C)."""
        if cls_id < 0 or cls_id >= self.num_classes:
            raise IndexError(f'class id {cls_id} out of range [0, {self.num_classes})')
        mask = self.valid_mask[cls_id]
        centers = self.centers[cls_id][mask]
        return centers

    def match_nearest(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return nearest prototypes for each feature.

        Args:
            features (Tensor): shape (N, C)
            labels   (Tensor): shape (N,) with class indices

        Returns:
            matched (Tensor): shape (N, C) matched prototype vectors (zeros if invalid)
            valid_mask (Tensor): shape (N,) bool, True when a prototype was found
            proto_indices (Tensor): shape (N,) index of selected prototype or -1
        """
        if features.size(0) == 0:
            device = labels.device if labels.is_cuda else features.device
            return (features.new_zeros(0, self.feature_dim),
                    torch.zeros(0, dtype=torch.bool, device=device),
                    torch.full((0,), -1, dtype=torch.long, device=device))

        device = features.device
        matched = torch.zeros(features.size(0), self.feature_dim, device=device)
        valid_mask = torch.zeros(features.size(0), dtype=torch.bool, device=device)
        proto_indices = torch.full((features.size(0),), -1, dtype=torch.long, device=device)

        for idx, (feat, label_tensor) in enumerate(zip(features, labels)):  # type: ignore
            label = int(label_tensor.item())
            if label < 0 or label >= self.num_classes:
                if self.default_to_identity:
                    matched[idx] = feat
                    valid_mask[idx] = True
                continue
            cls_mask = self.valid_mask[label]
            if not torch.any(cls_mask):
                if self.default_to_identity:
                    matched[idx] = feat
                    valid_mask[idx] = True
                continue
            cls_centers = self.centers[label][cls_mask].to(device)
            if self.normalize:
                feat_norm = F.normalize(feat.unsqueeze(0), dim=1)
                centers_norm = F.normalize(cls_centers, dim=1)
                similarities = torch.mm(feat_norm, centers_norm.t()).squeeze(0)
                best_idx = int(torch.argmax(similarities).item())
            else:
                distances = torch.cdist(feat.unsqueeze(0), cls_centers, p=2.0).squeeze(0)
                best_idx = int(torch.argmin(distances).item())
            matched[idx] = cls_centers[best_idx]
            valid_mask[idx] = True
            proto_indices[idx] = best_idx

        return matched.detach(), valid_mask, proto_indices

    def summary(self) -> Dict:
        """Return lightweight metadata for logging."""
        return {
            'num_classes': self.num_classes,
            'num_prototypes': self.num_prototypes,
            'feature_dim': self.feature_dim,
            'prototype_path': self.prototype_path,
            'metadata': self.metadata,
        }
