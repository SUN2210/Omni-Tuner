import torch
import torch.nn.functional as F

from mmdet.core import bbox2roi
from ..builder import HEADS, build_head, build_loss, build_prototype_bank
from .standard_roi_head import StandardRoIHead


@HEADS.register_module()
class PrototypeRoIHead(StandardRoIHead):
    def __init__(
        self,
        prototype_bank=None,
        alignment_loss=None,
        adversarial_loss=None,
        align_loss_weight=1.0,
        adv_loss_weight=1.0,
        freeze_bank=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.prototype_bank_cfg = prototype_bank
        self.alignment_loss_cfg = alignment_loss
        self.adversarial_loss_cfg = adversarial_loss
        self.align_loss_weight = align_loss_weight
        self.adv_loss_weight = adv_loss_weight
        self.freeze_bank = freeze_bank
        self.prototype_bank = None
        self.align_criterion = None
        self.adv_criterion = None

    def init_weights(self, pretrained):
        super().init_weights(pretrained)
        if self.prototype_bank_cfg is not None:
            self.prototype_bank = build_prototype_bank(self.prototype_bank_cfg)
            if self.freeze_bank:
                for param in self.prototype_bank.parameters():
                    param.requires_grad_(False)
        if self.alignment_loss_cfg is not None:
            self.align_criterion = build_loss(self.alignment_loss_cfg)
        if self.adversarial_loss_cfg is not None:
            self.adv_criterion = build_loss(self.adversarial_loss_cfg)

    def _bbox_forward_train(self, x, sampling_results, gt_bboxes, gt_labels, img_metas):
        rois = bbox2roi([res.bboxes for res in sampling_results])
        bbox_results = self._bbox_forward(x, rois)
        bbox_targets = self.bbox_head.get_targets(sampling_results, gt_bboxes, gt_labels, self.train_cfg)
        loss_bbox = self.bbox_head.loss(bbox_results['cls_score'], bbox_results['bbox_pred'], rois, *bbox_targets)
        bbox_results.update(loss_bbox=loss_bbox)
        if isinstance(bbox_results['loss_bbox'], dict):
            loss_dict = bbox_results['loss_bbox']
        else:
            loss_dict = bbox_results['loss_bbox']

        if self.prototype_bank is not None and (self.align_criterion is not None or self.adv_criterion is not None):
            pos_inds = torch.cat([res.pos_inds for res in sampling_results])
            if pos_inds.numel() > 0:
                bbox_feats = bbox_results['bbox_feats'][pos_inds]
                if bbox_feats.dim() == 4:
                    bbox_feats = F.adaptive_avg_pool2d(bbox_feats, 1).view(bbox_feats.size(0), -1)
                pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results])
                proto_feats, valid_mask, _ = self.prototype_bank.match_nearest(bbox_feats, pos_labels)
                if self.align_criterion is not None:
                    loss_align = self.align_criterion(bbox_feats, proto_feats, valid_mask)
                    loss_dict['loss_proto_align'] = loss_align * self.align_loss_weight
                if self.adv_criterion is not None:
                    loss_adv = self.adv_criterion(bbox_feats, proto_feats)
                    loss_dict['loss_proto_adv'] = loss_adv * self.adv_loss_weight
            else:
                zero = bbox_results['bbox_feats'].sum() * 0
                if self.align_criterion is not None:
                    loss_dict['loss_proto_align'] = zero
                if self.adv_criterion is not None:
                    loss_dict['loss_proto_adv'] = zero

        bbox_results['loss_bbox'] = loss_dict
        return bbox_results
