# 对偶模板监督损失（任务书 G）：堵死 C1-X 捷径
#
# 设计原理：同一张 IR 搜索图，正确模板 vs 错模板（donor）各做一次模板侧前向。
# 正确模板的输出必须在 GT 处高于错模板的全图最大响应，且错模板响应被整体压低。
# 捷径无解：不用模板身份，两条件输出相同 → margin 恒 0、损失不下降。
#
# 三项损失（权重写死，不许调）：
#   L_pos    = 现有 guided_focal（正确模板热图监督，train.py 中不变）
#   L_margin = relu(2.0 − (s_pos[GT中心] − max(s_neg)))，weight=1.0
#   L_neg    = mean(softplus(s_neg))，weight=0.1
#
# 配对规则（确定性）：同类（实例名前缀如 car_/pedestrian_）train 实例按名排序，
#   取本实例下一位（循环）为 donor；该类仅 1 实例时退化为全体 train 排序下一位
#   （cross_class_fallback=true）。donor 模板帧 = donor 全部模板帧升序中位。
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

CODE = Path(__file__).resolve().parents[1]


def instance_prefix(name):
    """实例名 → 类别前缀（car_008 → 'car_'）。"""
    return name.split('_')[0] + '_'


def build_pairing(inst_names):
    """确定性配对表：instance → {donor, same_class, fallback}。"""
    names = sorted(inst_names)
    groups = {}
    for n in names:
        groups.setdefault(instance_prefix(n), []).append(n)
    pairs = {}
    for n in names:
        g = groups[instance_prefix(n)]
        if len(g) > 1:
            i = g.index(n)
            donor = g[(i + 1) % len(g)]
            pairs[n] = {'donor': donor, 'same_class': True, 'fallback': False}
        else:
            i = names.index(n)
            donor = names[(i + 1) % len(names)]
            pairs[n] = {'donor': donor, 'same_class': False, 'fallback': True}
    return pairs


def load_donor_templates(root, pair_table, patch=224):
    """预加载每个实例的 donor 模板张量（与探针同取法：donor 全部模板帧升序中位）。
    返回 {instance: tensor(1,3,224,224)}。"""
    from data.cstloc_dataset import _to_tensor
    import cv2
    root = Path(root)
    out = {}
    for inst, info in pair_table.items():
        donor = info['donor']
        tmpl_dir = root / 'train' / donor / 'rgb' / 'template'
        frames = sorted(int(p.stem) for p in tmpl_dir.glob('*.jpg'))
        mid = frames[len(frames) // 2]
        p = tmpl_dir / f'{mid:06d}.jpg'
        img = cv2.imread(str(p))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (patch, patch), interpolation=cv2.INTER_LINEAR)
        out[inst] = _to_tensor(img).unsqueeze(0)   # (1,3,224,224)
    return out


class DualTemplateLoss(nn.Module):
    """L_margin + L_neg（L_pos 由 GuidedFocalMSELoss 在 train.py 计算）。"""

    def __init__(self, margin=2.0, neg_weight=0.1):
        super().__init__()
        self.margin = margin
        self.neg_weight = neg_weight

    def forward(self, logits_pos, logits_neg, heatmap):
        """
        logits_pos: (B,1,H,W) 正确模板输出
        logits_neg: (B,1,H,W) 错模板输出
        heatmap:    (B,1,H,W) GT 高斯热图（argmax = GT 中心，取整落格）
        Returns: (L_margin, L_neg)
        """
        s_pos = logits_pos.squeeze(1)                # (B,H,W)
        s_neg = logits_neg.squeeze(1)                # (B,H,W)
        gt_idx = heatmap.flatten(1).argmax(dim=1)    # (B,) 像素索引
        s_pos_gt = s_pos.flatten(1).gather(1, gt_idx.unsqueeze(1)).squeeze(1)
        max_neg = s_neg.flatten(1).max(dim=1).values
        margin_loss = F.relu(self.margin - (s_pos_gt - max_neg)).mean()
        neg_loss = F.softplus(s_neg).mean()
        return margin_loss, neg_loss
