# Correlation 调制器（本次主角）：模板 token 与 IR 特征的空间互相关
#
# 与 FiLM 的本质区别：
#   FiLM 给整张 IR 图相同的 gamma/beta（全局），无法区分同类多个目标；
#   Correlation 计算模板 token 与 IR 每个空间位置的相似度，得到空间相关图，
#   高相关处才被增强 —— 这才是"实例级空间匹配"，能应对多干扰物场景。
#
# 多模板天然支持：N 个模板的所有 token 一起参与相关，逐位置取最大相似度。
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseModulator
from .cmar import CMARReadout


class ProjectionAdapter(nn.Module):
    """任务书 1.9：冻结基底 + AdaptFormer 式瓶颈适配器残差（读出线冻结保护）。

    W(x) = W0·x + 0.1·up(ReLU(down(LN(x))))；W0 = 冻结 1×1 conv（基底权重由 train.py 按臂
    装载：A 臂 = S3 训练权重 / B 臂 = 随机正交阵 seed 42）；up 权重/偏置零初始化 → init ≡ W0
    恒等（G0a 位级核验）。LN 为逐空间位置的通道维 LayerNorm。
    思想移植自 MARCO references/repos/MARCO models/adapters.py（Apache-2.0；逐函数重写、
    零 import 耦合，出处注于报告）。
    """
    def __init__(self, c_in, c_out, factor=0.5, scale=0.1):
        super().__init__()
        self.w0 = nn.Conv2d(c_in, c_out, 1, bias=False)
        self.w0.requires_grad_(False)
        mid = max(int(c_out * factor), 1)
        self.ln = nn.LayerNorm(c_in)
        self.down = nn.Conv2d(c_in, mid, 1, bias=True)
        self.up = nn.Conv2d(mid, c_out, 1, bias=True)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)
        self.scale = scale

    def forward(self, x):
        y = self.w0(x)
        xn = x.permute(0, 2, 3, 1).contiguous()        # (B,H,W,C) 通道维 LN
        a = F.relu(self.down(self.ln(xn).permute(0, 3, 1, 2)))
        return y + self.scale * self.up(a)


class CorrLayer(nn.Module):
    """单层空间互相关调制。

    q = proj(IR特征)，k = proj(模板token)，逐位置对所有模板token取最大余弦相似度，
    得到相关图 corr(B,1,H,W)；再以 corr 作为空间注意力增强 IR 特征（保留残差）。

    实验 J：可选身份注入通路（use_id=True 时启用）——
      b_id = W_b·id_emb (B,C)、g_id = W_g·id_emb (B,C)；γ_b/γ_g (C,) 零初始化；
      attn = σ(scale·corr + bias + γ_b ⊙ b_id)；enhanced = ir_feat·attn ⊙ (1 + γ_g ⊙ g_id)。
      默认关闭（use_id=False 行为与注入前逐位一致，等价性测试保证）。
    """
    def __init__(self, channels, proj_dim=None, token_size=7, use_id=False, id_dim=256,
                 readout=None, cmar=False, cmar_mode='gate', cmar_gamma_init='zero',
                 cmar_direction='spatial'):
        super().__init__()
        # 实验 W：聚合读出。'max'（默认，行为逐位不变）| 'mlp'（sorted-MLP 学习式读出）。
        # 取值：显式参数 > 环境变量 CSTLOC_CORR_READOUT > 'max'（train.py 仿 CSTLOC_IR_ENCODER 传键）。
        self.readout = readout or os.environ.get('CSTLOC_CORR_READOUT', 'max')
        assert self.readout in ('max', 'mlp', 'res'), self.readout
        # 任务书 2.5：CMAR 门控注意力读出（默认关；开启时叠加在 max 读出上）。
        # CMARReadout 内部用 torch.empty + 专用 generator 初始化，不消耗全局 RNG，
        # 故构建位置不影响下游模块初始化（G0a/G0b 前置条件）。
        self.cmar = None
        self._last_attn_readout = None
        if cmar:
            assert self.readout == 'max', f'CMAR 叠加在 max 读出上，实际 readout={self.readout}'
            self.cmar = CMARReadout(channels, mode=cmar_mode,
                                    gamma_init=cmar_gamma_init, direction=cmar_direction)
        proj_dim = proj_dim or channels
        # 模板 token 池化到固定 token_size×token_size，界定相关计算量
        # （浅层模板图 28×28=784 token 过多，池化到 7×7=49 做区域匹配即可）
        self.token_pool = nn.AdaptiveAvgPool2d(token_size)
        # 任务书 1.9：冻结保护。env CSTLOC_PROJ_ADAPTER ∈ {s3, orth}（train.py 模型构建前设 env）
        # → proj_ir/proj_tmpl 换成 冻结基底+适配器 复合（基底权重构建后由 train.py 按臂装载）；
        # env 缺省 = 原 Conv2d 路径（行为逐位不变）。
        self.adapter_base = os.environ.get('CSTLOC_PROJ_ADAPTER')
        # 任务书 2.1：corr 图旁路开关（无适配器时也能取用 corr 做读出监督）。
        # env 缺省 → 不 stash，行为逐位不变。
        self.corr_bypass = bool(os.environ.get('CSTLOC_CORR_BYPASS'))
        if self.adapter_base:
            self.proj_ir = ProjectionAdapter(channels, proj_dim)
            self.proj_tmpl = ProjectionAdapter(channels, proj_dim)
        else:
            self.proj_ir = nn.Conv2d(channels, proj_dim, 1, bias=False)
            self.proj_tmpl = nn.Conv2d(channels, proj_dim, 1, bias=False)
        # 可学习温度与偏置：把 [-1,1] 的余弦相似度映射为注意力 logits
        self.scale = nn.Parameter(torch.tensor(10.0))
        self.bias = nn.Parameter(torch.tensor(0.0))
        # 融合：拼接 [原特征, 相关增强特征] → 卷回原通道
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.use_id = use_id
        if self.readout in ('mlp', 'res'):
            # sorted-MLP：排序后 token_size²-d 匹配分布 → 64 → 1（模板 N=1 时每位置匹配向量宽
            # = N×token_size²；W2 细粒度池化 token_size=16 → 256-d = 任务书 W 原设计的 256 维）
            self.readout_mlp = nn.Sequential(
                nn.Linear(token_size * token_size, 64), nn.ReLU(inplace=True),
                nn.Linear(64, 1))
            if self.readout == 'res':
                # W2 残差双读出：MLP 末层零初始化 → init 恒等于 max 读出（corr = max + 0），
                # 训练中只在分布形态携带增量信息处学习校正（G0b 核验）
                nn.init.zeros_(self.readout_mlp[2].weight)
                nn.init.zeros_(self.readout_mlp[2].bias)
        if use_id:
            self.w_b = nn.Linear(id_dim, channels, bias=False)
            self.w_g = nn.Linear(id_dim, channels, bias=False)
            self.gamma_b = nn.Parameter(torch.zeros(channels))   # 零初始化 → 起点等价 s1
            self.gamma_g = nn.Parameter(torch.zeros(channels))

    def _base_corr(self, ir_feat, tmpl_map):
        """冻结基底（w0）通路的 corr = max-over-tokens 余弦（须在 no_grad 内调用；
        1.9 自蒸馏目标 = 适配器 corr 向基底 corr 回归）。"""
        B = ir_feat.shape[0]
        N = tmpl_map.shape[1]
        q = F.normalize(self.proj_ir.w0(ir_feat), dim=1)
        k = self.proj_tmpl.w0(tmpl_map.flatten(0, 1))
        k = F.normalize(self.token_pool(k), dim=1)
        D = k.shape[1]
        k = k.view(B, N, D, -1)
        k = k.permute(0, 2, 1, 3).reshape(B, D, N * k.shape[-1])
        sim = torch.einsum('bdm,bdn->bmn', q.flatten(2), k)
        return sim.max(dim=2).values.view(B, 1, ir_feat.shape[2], ir_feat.shape[3])

    def forward(self, ir_feat, tmpl_map, id_emb=None):
        """
        ir_feat:  (B, C, H, W)
        tmpl_map: (B, N, C, h, w)
        id_emb:   (B, id_dim) 身份嵌入（use_id=True 且非 None 时注入）
        """
        B, C, H, W = ir_feat.shape
        N = tmpl_map.shape[1]

        q = self.proj_ir(ir_feat)                          # (B, D, H, W)
        q = F.normalize(q, dim=1)

        k = self.proj_tmpl(tmpl_map.flatten(0, 1))         # (B*N, D, h, w)
        k = self.token_pool(k)                             # (B*N, D, ts, ts)
        k = F.normalize(k, dim=1)
        D = k.shape[1]
        k = k.view(B, N, D, -1)                            # (B, N, D, h*w)
        k = k.permute(0, 2, 1, 3).reshape(B, D, N * k.shape[-1])  # (B, D, N*h*w)

        # 逐 IR 位置与所有模板 token 的余弦相似度
        q_flat = q.flatten(2)                              # (B, D, H*W)
        sim = torch.einsum('bdm,bdn->bmn', q_flat, k)      # (B, H*W, N*h*w)
        if self.readout in ('mlp', 'res'):
            # sorted-MLP：沿模板 token 维降序排列（排列不变）→ MLP 读匹配分布形态
            K = sim.shape[2]
            assert K == self.readout_mlp[0].in_features,                 f'mlp 读出要求模板 token 数 {self.readout_mlp[0].in_features}，实际 {K}'
            s_sorted = torch.sort(sim, dim=2, descending=True).values
            if self.readout == 'res':
                # W2 残差双读出：max 保底 easy 单峰先验 + MLP 学分布校正（init ≡ max，G0b 核验）
                corr = sim.max(dim=2).values + self.readout_mlp(s_sorted).squeeze(-1)
            else:
                corr = self.readout_mlp(s_sorted).squeeze(-1)  # (B, H*W)
        else:
            corr = sim.max(dim=2).values                   # (B, H*W)
        if self.cmar is not None:
            # 任务书 2.5：corr(p) = max + γ·attn_readout(p)；γ=0 时与 max 读出逐位一致（G0a）。
            # 消融格①（mode='replace'）：corr = attn_readout 直接替换 max 读出。
            # 注意：k 即相关计算实际使用的模板 token（proj+normalize 后，(B,D,T)）。
            attn_readout = self.cmar(sim, ir_feat, k)      # (B, H*W)
            self._last_attn_readout = attn_readout         # 探针缓存（取用后 reshape (B,H,W)）
            corr = attn_readout if self.cmar.mode == 'replace' \
                else corr + self.cmar.gamma * attn_readout
        corr = corr.view(B, 1, H, W)
        if self.adapter_base or self.corr_bypass:
            # 任务书 1.9/2.1：corr 图旁路输出（图内，train.py 读出监督用；不经 localizer.py 改动）
            self._last_corr = corr
            if self.adapter_base:
                with torch.no_grad():                      # 冻结基底通路 corr（自蒸馏目标）
                    self._last_corr_base = self._base_corr(ir_feat, tmpl_map)

        attn = torch.sigmoid(self.scale * corr + self.bias)   # (B,1,H,W)
        enhanced = ir_feat * attn
        if self.use_id and id_emb is not None:
            b_id = self.w_b(id_emb)                        # (B,C)
            g_id = self.w_g(id_emb)
            attn = torch.sigmoid(self.scale * corr + self.bias
                                 + self.gamma_b.view(1, C, 1, 1) * b_id.unsqueeze(-1).unsqueeze(-1))
            enhanced = ir_feat * attn * (1 + self.gamma_g.view(1, C, 1, 1)
                                         * g_id.unsqueeze(-1).unsqueeze(-1))
        return self.fuse(torch.cat([ir_feat, enhanced], dim=1))


class CorrelationModulator(BaseModulator):
    needs = 'map'

    def __init__(self, use_id=False, id_dim=256, token_size=None,
                 cmar=False, cmar_mode='gate', cmar_gamma_init='zero',
                 cmar_direction='spatial'):
        super().__init__()
        self.use_id = use_id
        # 实验 W2：token_size 透传。取值：显式参数 > env CSTLOC_CORR_TOKEN_SIZE > 7。
        # env 须在模型构建前设置（train.py 仿 CSTLOC_CORR_READOUT 传键——W 的 env 落点教训）。
        if token_size is None:
            token_size = int(os.environ.get('CSTLOC_CORR_TOKEN_SIZE', 7))
        cm = dict(cmar=cmar, cmar_mode=cmar_mode, cmar_gamma_init=cmar_gamma_init,
                  cmar_direction=cmar_direction)
        self.corr_x3 = CorrLayer(128, use_id=use_id, id_dim=id_dim, token_size=token_size, **cm)
        self.corr_x4 = CorrLayer(256, use_id=use_id, id_dim=id_dim, token_size=token_size, **cm)
        self.corr_x5 = CorrLayer(512, use_id=use_id, id_dim=id_dim, token_size=token_size, **cm)

    def forward(self, ir_feats, tmpl_vecs, tmpl_maps, id_emb=None):
        x3, x4, x5 = ir_feats
        m3, m4, m5 = tmpl_maps
        x3 = self.corr_x3(x3, m3, id_emb)
        x4 = self.corr_x4(x4, m4, id_emb)
        x5 = self.corr_x5(x5, m5, id_emb)
        return x3, x4, x5
