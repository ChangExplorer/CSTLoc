# 身份专家（实验 J 阶段 2 / 实验 J2 阶段 2）：冻结专家，输入 DINOv2 特征 → id_emb (B,256)
#
# 两种形态（由 checkpoint 的 arch 字段区分）：
#   spatial（实验 J，已归档）：加载 j_expert_pretrain/expert_head.pt 的 SpatialHead 权重，
#     输入 dinov2 intermediate_taps（reshape=True，256 patch token）maps list → (B,256)
#   linear 系（实验 J2）：arch 为 'linear' / 'linear-hse' 时，结构 = fc_g(2048→256)+bn_g
#     （从 head_<variant>.pt 加载，cls_g 及 fc_l/bn_l/局部分类器丢弃），forward 输入 (B,2048)
#     （dinov2 intermediate_gap 输出）→ post-BN 256-d
# 冻结 + eval（沿用现有写法）。
import torch
import torch.nn as nn
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
DEFAULT_CKPT = CODE / 'checkpoints' / 'j_expert_pretrain' / 'expert_head.pt'
DEFAULT_TAPS = [15, 23]


class IdentityExpert(nn.Module):
    def __init__(self, ckpt_path=None, taps=DEFAULT_TAPS, emb=256):
        super().__init__()
        ckpt_path = Path(ckpt_path) if ckpt_path else DEFAULT_CKPT
        self.taps = taps
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
