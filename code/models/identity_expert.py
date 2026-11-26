# 随机身份专家头：冻结的 fc_g(2048→256) + BatchNorm1d。
# 输入 DINOv2 模板特征的 intermediate GAP (B,2048)，输出身份嵌入 (B,256)。
#
# 生成方式（不随仓库分发权重，完全由代码确定性生成）：
#   在 fork 出的 RNG 中用固定整数 seed 初始化 fc_g = nn.Linear(2048, 256)，
#   bn_g 保持 PyTorch 默认初始化；生成后整体冻结（requires_grad=False、eval）。
# 任何固定整数都可用作 seed；只要求 locator 训练与评测共用同一个 seed。
# 默认 0，可用环境变量 CSTLOC_EXPERT_SEED 覆盖。
import os

import torch
import torch.nn as nn

DEFAULT_EMB = 256


class IdentityExpert(nn.Module):
    arch = 'linear-random'

    def __init__(self, ckpt_path='random', taps=None, emb=DEFAULT_EMB):
        super().__init__()
        if str(ckpt_path).lower() not in ('random', 'linear-random'):
            raise ValueError("the released system uses a code-generated random expert; "
                             "pass identity_expert='random'")
        seed = int(os.environ.get('CSTLOC_EXPERT_SEED', '0'))
        # 先在全局 RNG 下构建模块（保持与其他组件一致的初始化消耗顺序），
        # 再在 fork 内用固定 seed 覆盖 fc_g 权重。
        self.fc_g = nn.Linear(2048, emb)
        self.bn_g = nn.BatchNorm1d(emb)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            ref_fc = nn.Linear(2048, emb)
        self.fc_g.load_state_dict(ref_fc.state_dict())
        self.bn_g.bias.requires_grad_(False)
        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        """x (B,2048) DINOv2 intermediate GAP → (B,256) 身份嵌入。"""
        return self.bn_g(self.fc_g(x))
