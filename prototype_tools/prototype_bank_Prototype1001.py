from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


class PrototypeBankPrototype1001:
    def __init__(self, prototype_dict: Dict[int, Dict[str, torch.Tensor]], device: torch.device,
                 momentum: float = 0.0):
        self.device = device
        self.momentum = momentum
        self.centers: Dict[int, torch.Tensor] = {}
        self.counts: Dict[int, torch.Tensor] = {}
        self.feature_dim: Optional[int] = None
        for cls_id, entry in prototype_dict.items():
            centers = entry.get('centers')
            if centers is None:
                raise ValueError(f'class {cls_id} missing "centers" field in prototype dict')
            centers_tensor = torch.as_tensor(centers, dtype=torch.float32, device=device)
            self.centers[int(cls_id)] = centers_tensor
            if self.feature_dim is None and centers_tensor.numel() > 0:
                self.feature_dim = int(centers_tensor.size(1))
            counts = entry.get('counts')
            if counts is not None:
                self.counts[int(cls_id)] = torch.as_tensor(counts, dtype=torch.float32, device=device)
            else:
                self.counts[int(cls_id)] = torch.ones(centers_tensor.size(0), device=device)

    def get_all(self, cls_id: int) -> torch.Tensor:
        return self.centers[int(cls_id)]

    def sample_center(self, cls_id: int, temperature: float = 1.0) -> Tuple[torch.Tensor, int]:
        centers = self.centers[int(cls_id)]
        counts = self.counts[int(cls_id)]
        weights = counts / counts.sum()
        if temperature != 1.0:
            weights = torch.softmax(torch.log(weights + 1e-6) / temperature, dim=0)
        idx = torch.multinomial(weights, 1)
        index = idx.item()
        return centers[index], index

    def pairwise_distance(self, features: torch.Tensor, cls_ids: torch.Tensor,
                           normalize: bool) -> torch.Tensor:
        dists = []
        for feat, cid in zip(features, cls_ids):
            centers = self.get_all(int(cid.item()))
            if normalize:
                feat = F.normalize(feat.unsqueeze(0), dim=1)
                centers = F.normalize(centers, dim=1)
                dist = 1 - torch.matmul(feat, centers.t()).squeeze(0)
            else:
                dist = torch.cdist(feat.unsqueeze(0), centers).squeeze(0)
            dists.append(dist)
        return torch.stack(dists)

    def find_best_match(self, feature: torch.Tensor, normalize: bool) -> Tuple[Optional[int], Optional[int], Optional[torch.Tensor], float]:
        """Return the best matching prototype across all classes.

        Args:
            feature (Tensor): Feature vector with shape (C,).
            normalize (bool): Whether to apply L2 normalization before matching.

        Returns:
            Tuple containing selected class id, prototype index, prototype tensor, and similarity score.
            If no prototypes are registered, returns (None, None, None, float('-inf')).
        """
        if not self.centers:
            return None, None, None, float('-inf')

        if feature.dim() == 1:
            feat = feature.unsqueeze(0)
        else:
            feat = feature
        if normalize:
            feat = F.normalize(feat, dim=1)

        best_cls: Optional[int] = None
        best_idx: Optional[int] = None
        best_center: Optional[torch.Tensor] = None
        best_score = float('-inf')

        for cls_id, centers in self.centers.items():
            centers_tensor = centers
            if centers_tensor.numel() == 0:
                continue
            if normalize:
                centers_tensor = F.normalize(centers_tensor, dim=1)
                scores = torch.mm(feat, centers_tensor.t()).squeeze(0)
                score_val, score_idx = torch.max(scores, dim=0)
                score_float = float(score_val.item())
            else:
                dists = torch.cdist(feat, centers_tensor).squeeze(0)
                score_val, score_idx = torch.min(dists, dim=0)
                score_float = -float(score_val.item())  # convert to similarity-like score
            if score_float > best_score:
                best_score = score_float
                best_cls = cls_id
                best_idx = int(score_idx.item())
                best_center = centers[best_idx]

        return best_cls, best_idx, best_center, best_score

    def update(self, cls_id: int, features: torch.Tensor) -> None:
        if self.momentum <= 0.0 or features.numel() == 0:
            return
        features = features.detach().to(self.device)
        centers = self.centers[int(cls_id)]
        mean = features.mean(dim=0, keepdim=True)
        self.centers[int(cls_id)] = (1 - self.momentum) * centers + self.momentum * mean
        self.counts[int(cls_id)] += features.size(0)

    def to(self, device: torch.device):
        for key in list(self.centers.keys()):
            self.centers[key] = self.centers[key].to(device)
            self.counts[key] = self.counts[key].to(device)
        self.device = device
        return self

    def get_feature_dim(self) -> Optional[int]:
        return self.feature_dim
