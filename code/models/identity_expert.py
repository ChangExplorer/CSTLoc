# 身份专家（实验 J 阶段 2 / 实验 J2 阶段 2）：冻结专家，输入 DINOv2 特征 → id_emb (B,256)
#
# 两种形态（由 checkpoint 的 arch 字段区分）：
#   spatial（实验 J，已归档）：加载 j_expert_pretrain/expert_head.pt 的 SpatialHead 权重，
#     输入 dinov2 intermediate_taps（reshape=True，256 patch token）maps list → (B,256)
#   linear 系（实验 J2）：arch 为 'linear' / 'linear-hse' 时，结构 = fc_g(2048→256)+bn_g
#     （从 head_<variant>.pt 加载，cls_g 及 fc_l/bn_l/局部分类器丢弃），forward 输入 (B,2048)
#     （dinov2 intermediate_gap 输出）→ post-BN 256-d
# 冻结 + eval（沿用现有写法）。
import os

import torch
import torch.nn as nn
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
DEFAULT_CKPT = CODE / 'checkpoints' / 'j_expert_pretrain' / 'expert_head.pt'
DEFAULT_TAPS = [15, 23]


class IdentityExpert(nn.Module):
    def __init__(self, ckpt_path=None, taps=DEFAULT_TAPS, emb=256):
        super().__init__()
        ckpt_path = str(ckpt_path) if ckpt_path else str(DEFAULT_CKPT)
        self.taps = taps
        # 训练/评测时按固定种子生成随机专家头，不再随包分发 5MB 权重。
        # 模块的构建方式与历史文件分支相同（消耗相同全局 RNG），随机头权重再用固定种子覆盖，
        # 因此定位器后续初始化顺序与历史发布权重逐位一致。
        if ckpt_path.lower() in ('random', 'linear-random'):
            seed = int(os.environ.get('CSTLOC_EXPERT_SEED', '123'))
            # 先按历史文件分支的同一方式在全局 RNG 下构建模块，消耗相同的随机数；
            # 再在 fork 内用固定种子覆盖权重。这样后续 id_null 等初始化与历史逐位一致，
            # 同时随机头本身与历史 head_random.pt 逐位一致。
            self.fc_g = nn.Linear(2048, emb)
            self.bn_g = nn.BatchNorm1d(emb)
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                ref_fc = nn.Linear(2048, emb)
            self.fc_g.load_state_dict(ref_fc.state_dict())
            self.bn_g.bias.requires_grad_(False)
            self.arch = 'linear-random'
            self.eval()
            for p in self.parameters():
                p.requires_grad_(False)
            return
        ckpt_path = Path(ckpt_path)
        state = torch.load(ckpt_path, map_location='cpu')
        arch = state.get('arch', 'spatial') if isinstance(state, dict) else 'spatial'
        self.arch = str(arch)
        head = state['head'] if isinstance(state, dict) and 'head' in state else state
        if self.arch.startswith('linear'):
            # J2 线性专家：fc_g(2048→256) + bn_g（BN 偏置冻结，与预训练一致）
            self.fc_g = nn.Linear(2048, emb)
            self.bn_g = nn.BatchNorm1d(emb)
            self.bn_g.bias.requires_grad_(False)
            keep = {k: v for k, v in head.items() if k.startswith(('fc_g.', 'bn_g.'))}
            self.load_state_dict(keep, strict=False)
        else:
            self.conv_taps = nn.ModuleList([nn.Conv2d(1024, emb, 1) for _ in taps])
            self.tf = nn.TransformerEncoderLayer(d_model=emb, nhead=4, dim_feedforward=1024,
                                                 batch_first=True)
            self.neck = nn.BatchNorm1d(emb)
            self.neck.bias.requires_grad_(False)
            for k in list(head.keys()):
                if k.startswith('cls.'):
                    head.pop(k)
            self.load_state_dict(head, strict=False)
        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        """linear 系：x (B,2048) GAP → (B,256)；
        spatial：maps_list [(B,1024,h,w), ...]（dinov2 taps reshape=True）→ (B,256)"""
        if self.arch.startswith('linear'):
            return self.bn_g(self.fc_g(x))
        toks = []
        for m, conv in zip(x, self.conv_taps):
            p = conv(m)
            B, C, h, w = p.shape
            toks.append(p.flatten(2).permute(0, 2, 1))
        seq = torch.cat(toks, dim=1)
        seq = self.tf(seq)
        e = seq.mean(dim=1)
        return self.neck(e)
