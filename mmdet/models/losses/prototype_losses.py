import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES
from ..utils.grl import GradientReversal


@LOSSES.register_module()
class PrototypeAlignmentLoss(nn.Module):
    def __init__(self, loss_weight=1.0, normalize=True, metric='l2'):
        super().__init__()
        self.loss_weight = loss_weight
        self.normalize = normalize
        self.metric = metric

    def forward(self, features, prototypes, valid_mask):
        if features.numel() == 0 or not valid_mask.any():
            return features.new_tensor(0.)
        feats = features[valid_mask]
        protos = prototypes[valid_mask]
        if self.normalize:
            feats = F.normalize(feats, dim=1)
            protos = F.normalize(protos, dim=1)
        if self.metric == 'cosine':
            # cosine alignment via built-in API
            loss = 1 - F.cosine_similarity(feats, protos, dim=1)
        elif self.metric == 'l2':
            # sum of squared Euclidean distance per sample
            loss = (feats - protos).pow(2).sum(dim=1)
        elif self.metric == 'mse':
            # mean squared error summed over feature dims per sample
            loss = F.mse_loss(feats, protos, reduction='none').sum(dim=1)

        else:
            raise ValueError(f'Unsupported metric: {self.metric}')
        return loss.mean() * self.loss_weight


@LOSSES.register_module()
class PrototypeAdversarialLoss(nn.Module):
    def __init__(self, in_dim, hidden_dim=256, num_layers=2, loss_weight=1.0,
                 gradient_reverse_weight=1.0, detach_prototypes=True):
        super().__init__()
        layers = []  # 初始化判别器线性层列表
        last_dim = in_dim  # 记录当前层的输入特征维度
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(last_dim, hidden_dim))  # 堆叠全连接层以增加判别器容量
            layers.append(nn.ReLU(inplace=True))  # 使用 ReLU 激活保持非线性
            last_dim = hidden_dim  # 下一层输入维度更新为隐藏维度
        layers.append(nn.Linear(last_dim, 1))  # 最后一层输出单通道 logit，用于二分类
        self.discriminator = nn.Sequential(*layers)  # 将层列表封装为顺序判别器
        self.loss_weight = loss_weight  # 存储损失权重，方便外部缩放
        self.gradient_reversal = GradientReversal(gradient_reverse_weight)  # 梯度反转模块，控制对抗强度
        self.sigmoid = nn.Sigmoid()  # 备用 Sigmoid（保留以兼容潜在调用）
        self.detach_prototypes = detach_prototypes  # 是否阻断原型特征的反向传播

    def forward(self, target_features, prototype_features):
        if target_features.numel() == 0:
            return target_features.new_tensor(0.)  # 无目标域特征时直接返回零损失
        target_features = self.gradient_reversal(target_features)  # 对目标域特征施加梯度反转，实现对抗更新
        logits_t = self.discriminator(target_features)  # 判别器尝试区分这些特征（应靠近 1）
        labels_t = torch.ones_like(logits_t)  # 目标域标签标记为 1（判别器认为是“目标域”）
        loss_t = F.binary_cross_entropy_with_logits(logits_t, labels_t)  # 计算目标域分支的 BCE 损失
        if prototype_features is not None and prototype_features.numel() > 0:
            if self.detach_prototypes:
                prototype_features = prototype_features.detach()  # 默认截断梯度，保持原型不被判别器更新
            logits_p = self.discriminator(prototype_features)  # 判别器判断源域原型（应靠近 0）
            labels_p = torch.zeros_like(logits_p)  # 源域原型标签设为 0
            loss_p = F.binary_cross_entropy_with_logits(logits_p, labels_p)  # 计算原型分支的 BCE 损失
        else:
            loss_p = target_features.new_tensor(0.)  # 若没有原型特征，原型分支贡献为 0
        return (loss_t + loss_p) * 0.5 * self.loss_weight  # 两个分支求平均后再乘以权重
