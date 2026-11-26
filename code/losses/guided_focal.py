# 引导式 Focal MSE 损失（含干扰物抑制项）
#
# 迁移自原项目。在 Correlation 架构下抑制项重新生效：
# 空间相关能对不同位置给不同响应，因此"压低干扰物、保留目标"物理上可行
# （FiLM 全局调制对同类目标响应相同，抑制项无从发力——这正是要对比的点）。
import torch
import torch.nn as nn


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class GuidedFocalMSELoss(nn.Module):
    """
    total = focal_mse(前景加权) + suppress_weight * suppression

    Args:
        alpha:           Focal 加权指数
        pos_weight:      前景像素相对背景的额外权重
        suppress_weight: 干扰物抑制项权重
    """
    def __init__(self, alpha=2.0, pos_weight=10.0, suppress_weight=5.0):
        super().__init__()
        self.alpha = alpha
        self.pos_weight = pos_weight
        self.suppress_weight = suppress_weight

    def forward(self, pred, target, suppression_mask):
        p = torch.sigmoid(pred)
        diff = torch.abs(p - target)
        mse = (p - target) ** 2
        focal_w = (1.0 + diff) ** self.alpha
        fg_w = 1.0 + (self.pos_weight - 1.0) * target
        focal_mse = (focal_w * fg_w * mse).mean()

        mask_sum = suppression_mask.sum().clamp(min=1.0)
        suppression = (suppression_mask * p ** 2).sum() / mask_sum
        return focal_mse + self.suppress_weight * suppression
