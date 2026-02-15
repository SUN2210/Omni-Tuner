from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, bias_init_with_prob, normal_init
from mmcv.runner import force_fp32

from mmdet.core import bbox2roi, images_to_levels, multi_apply

from ..builder import HEADS, build_roi_extractor
from .anchor_head import AnchorHead


@HEADS.register_module()
class RetinaHead(AnchorHead):
    r"""An anchor-based head used in `RetinaNet
    <https://arxiv.org/pdf/1708.02002.pdf>`_.

    The head contains two subnetworks. The first classifies anchor boxes and
    the second regresses deltas for the anchors.

    Example:
        >>> import torch
        >>> self = RetinaHead(11, 7)
        >>> x = torch.rand(1, 7, 32, 32)
        >>> cls_score, bbox_pred = self.forward_single(x)
        >>> # Each anchor predicts a score for each class except background
        >>> cls_per_anchor = cls_score.shape[1] / self.num_anchors
        >>> box_per_anchor = bbox_pred.shape[1] / self.num_anchors
        >>> assert cls_per_anchor == (self.num_classes)
        >>> assert box_per_anchor == 4
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 stacked_convs=4,
                 conv_cfg=None,
                 norm_cfg=None,
                 anchor_generator=dict(
                     type='AnchorGenerator',
                     octave_base_scale=4,
                     scales_per_octave=3,
                     ratios=[0.5, 1.0, 2.0],
                     strides=[8, 16, 32, 64, 128]),
                 **kwargs):
        self.stacked_convs = stacked_convs
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        super(RetinaHead, self).__init__(
            num_classes,
            in_channels,
            anchor_generator=anchor_generator,
            **kwargs)
        self.prototype_manager = None
        self.prototype_roi_extractor = None
        self._prototype_cache_key: Optional[Tuple[Tuple[int, ...], Tuple[int, ...]]] = None

    def _init_layers(self):
        """Initialize layers of the head."""
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
        self.retina_cls = nn.Conv2d(
            self.feat_channels,
            self.num_anchors * self.cls_out_channels,
            3,
            padding=1)
        self.retina_reg = nn.Conv2d(
            self.feat_channels, self.num_anchors * 4, 3, padding=1)

    def init_weights(self):
        """Initialize weights of the head."""
        for m in self.cls_convs:
            normal_init(m.conv, std=0.01)
        for m in self.reg_convs:
            normal_init(m.conv, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.retina_cls, std=0.01, bias=bias_cls)
        normal_init(self.retina_reg, std=0.01)

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def loss(self,
             cls_scores,
             bbox_preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None,
             feats=None):
        """Compute losses of the head with optional prototype adaptation."""
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.anchor_generator.num_levels

        device = cls_scores[0].device

        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas, device=device)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        need_sampling_results = feats is not None and getattr(self, 'prototype_manager', None) is not None
        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels,
            return_sampling_results=need_sampling_results)
        if cls_reg_targets is None:
            return None

        (labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, num_total_pos, num_total_neg) = cls_reg_targets[:6]
        sampling_results_list = cls_reg_targets[6] if need_sampling_results else None
        num_total_samples = (
            num_total_pos + num_total_neg if self.sampling else num_total_pos)

        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        concat_anchor_list = []
        for i in range(len(anchor_list)):
            concat_anchor_list.append(torch.cat(anchor_list[i]))
        all_anchor_list = images_to_levels(concat_anchor_list, num_level_anchors)

        losses_cls, losses_bbox = multi_apply(
            self.loss_single,
            cls_scores,
            bbox_preds,
            all_anchor_list,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_samples=num_total_samples)

        losses = dict(loss_cls=losses_cls, loss_bbox=losses_bbox)

        if need_sampling_results and sampling_results_list is not None:
            proto_losses, proto_logs = self._compute_prototype_losses(
                feats, sampling_results_list, gt_labels, gt_bboxes)
            for key, value in proto_losses.items():
                losses[key] = value
            for key, value in proto_logs.items():
                losses[f'log_{key}'] = value

        return losses

    def forward_single(self, x):
        """Forward feature of a single scale level.

        Args:
            x (Tensor): Features of a single scale level.

        Returns:
            tuple:
                cls_score (Tensor): Cls scores for a single scale level
                    the channels number is num_anchors * num_classes.
                bbox_pred (Tensor): Box energies / deltas for a single scale
                    level, the channels number is num_anchors * 4.
        """
        cls_feat = x
        reg_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)
        cls_score = self.retina_cls(cls_feat)
        bbox_pred = self.retina_reg(reg_feat)
        return cls_score, bbox_pred

    def forward_train(self,
                      x,
                      img_metas,
                      gt_bboxes,
                      gt_labels=None,
                      gt_bboxes_ignore=None,
                      proposal_cfg=None,
                      **kwargs):
        cls_scores, bbox_preds = self(x)
        losses = self.loss(
            cls_scores,
            bbox_preds,
            gt_bboxes,
            gt_labels,
            img_metas,
            gt_bboxes_ignore=gt_bboxes_ignore,
            feats=x)
        if proposal_cfg is None:
            return losses
        # RetinaNet does not output proposals; maintain interface.
        return losses, None

    def _infer_featmap_strides(self) -> Tuple[int, ...]:
        strides = getattr(self, 'featmap_strides', None)
        if strides is not None:
            return tuple(int(s) for s in strides)
        if hasattr(self, 'anchor_generator'):
            ag_strides = getattr(self.anchor_generator, 'strides', None)
            if ag_strides is not None:
                return tuple(
                    int(s[0] if isinstance(s, (list, tuple)) else s)
                    for s in ag_strides)
        raise AttributeError('Unable to infer feature map strides for RetinaHead prototype extraction.')

    def _get_prototype_roi_extractor(self, feats: Sequence[torch.Tensor]):
        strides = self._infer_featmap_strides()
        channel_signature = tuple(int(feat.shape[1]) for feat in feats[:len(strides)])
        cache_key = (strides, channel_signature)

        if self.prototype_roi_extractor is None or self._prototype_cache_key != cache_key:
            extractor_cfg = dict(
                type='SingleRoIExtractor',
                roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
                out_channels=channel_signature[0],
                featmap_strides=list(strides),
            )
            self.prototype_roi_extractor = build_roi_extractor(extractor_cfg)
            self._prototype_cache_key = cache_key

        device = feats[0].device
        self.prototype_roi_extractor = self.prototype_roi_extractor.to(device)
        self.prototype_roi_extractor.train(self.training)
        return self.prototype_roi_extractor

    def _gather_positive_samples(
            self,
            sampling_results: List,
            gt_labels: Sequence[torch.Tensor],
            gt_bboxes: Sequence[torch.Tensor]) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        prefer_gt = getattr(getattr(self, 'prototype_manager', None), 'prefer_gt_boxes', False)

        roi_sources: List[torch.Tensor] = []
        label_tensors: List[torch.Tensor] = []

        for img_idx, res in enumerate(sampling_results):
            if prefer_gt:
                if res.pos_assigned_gt_inds.numel() == 0:
                    continue
                unique_gt = torch.unique(res.pos_assigned_gt_inds)
                unique_gt = unique_gt[unique_gt >= 0]
                if unique_gt.numel() == 0:
                    continue
                img_gt_bboxes = gt_bboxes[img_idx]
                img_gt_labels = gt_labels[img_idx]
                if img_gt_bboxes.numel() == 0 or img_gt_labels.numel() == 0:
                    continue
                roi_sources.append(img_gt_bboxes[unique_gt])
                label_tensors.append(img_gt_labels[unique_gt])
            else:
                if res.pos_bboxes.numel() == 0:
                    continue
                roi_sources.append(res.pos_bboxes)
                labels = res.pos_gt_labels
                if labels is None:
                    img_gt_labels = gt_labels[img_idx] if gt_labels is not None else None
                    if img_gt_labels is None or img_gt_labels.numel() == 0:
                        continue
                    labels = img_gt_labels[res.pos_assigned_gt_inds]
                label_tensors.append(labels)

        if not roi_sources or not label_tensors:
            return None, None

        rois = bbox2roi(roi_sources)
        if rois.numel() == 0:
            return None, None

        labels_cat = torch.cat(label_tensors, dim=0)
        return rois, labels_cat

    def _compute_prototype_losses(
        self,
        feats: Sequence[torch.Tensor],
        sampling_results: List,
        gt_labels: Sequence[torch.Tensor],
        gt_bboxes: Sequence[torch.Tensor]) -> Tuple[dict, dict]:
        if getattr(self, 'prototype_manager', None) is None:
            return {}, {}
        if not sampling_results:
            return {}, {}
        rois, labels = self._gather_positive_samples(sampling_results, gt_labels, gt_bboxes)
        if rois is None or labels is None or labels.numel() == 0:
            return {}, {}

        roi_extractor = self._get_prototype_roi_extractor(feats)
        device = feats[0].device
        roi_feats = roi_extractor(feats[:roi_extractor.num_inputs], rois.to(device))
        pooled = roi_feats.mean(dim=[2, 3])
        labels = labels.to(device)

        proto_losses, proto_stats = self.prototype_manager.compute_losses(pooled, labels)
        log_tensors = {}
        if proto_stats:
            for key, value in proto_stats.items():
                log_tensors[key] = pooled.new_tensor(value)
        return proto_losses, log_tensors

    def init_weights(self):
        """Initialize weights of the head."""
        for m in self.cls_convs:
            normal_init(m.conv, std=0.01)
        for m in self.reg_convs:
            normal_init(m.conv, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.retina_cls, std=0.01, bias=bias_cls)
        normal_init(self.retina_reg, std=0.01)
