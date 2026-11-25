# 简单融合调制器（消融用）：双分支特征 concat + 1x1 投影，保持跨模态连通
# 严格不包含 CorrLayer 的空间相关计算；用于“简单融合”行，不是把网络拆散。
import torch
import torch.nn as nn

from .base import BaseModulator


class SimpleFusionModulator(BaseModulator):
    needs = 'vec'

    def __init__(self):
        super().__init__()
        chs = (128, 256, 512)
        self.proj = nn.ModuleList([nn.Linear(c, c) for c in chs])
        self.fuse = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c * 2, c, 1, bias=False),
                nn.BatchNorm2d(c),
                nn.ReLU(inplace=True),
            ) for c in chs
        ])

    def forward(self, ir_feats, tmpl_vecs, tmpl_maps, id_emb=None):
        outs = []
        for x, v, p, f in zip(ir_feats, tmpl_vecs, self.proj, self.fuse):
            # v: (B,N,C) → 模板全局向量平均 → 1x1 线性投影 → 广播到空间
            cond = p(v.mean(dim=1)).unsqueeze(-1).unsqueeze(-1)
            cond = cond.expand(-1, -1, x.shape[-2], x.shape[-1])
            outs.append(f(torch.cat([x, cond], dim=1)))
        return tuple(outs)
