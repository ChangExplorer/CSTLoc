# 任务书 2.5：CMAR（Cross-template Memory Attention Readout）——零初始化门控注意力读出
#
# 结构（对 CorrLayer 的单点改动，其余路径逐位不变）：
#   原读出：corr(p) = max_t S(p, t)
#   新读出：corr(p) = max_t S(p, t) + γ_ℓ · attn_readout(p)
#   attn_readout(p) = W_o · concat_h[ Σ_t α^h_{p,t} · S(p, t) ]
#   α^h_{p,·} = softmax_t( (W_q f_p)^h · (W_k token_t)^h / √d_h )
#   q_p = W_q·f_p    f_p = 该尺度 IR 场景特征在 p 处的通道向量（C 维）
#   k_t = W_k·token_t  token_t = 相关计算实际使用的模板 token（proj + L2 归一化后）
#
# 设计约束（任务书 §3/§4 + lead 设计决定，均写入报告）：
#   - 单层注意力、8 头、hidden = 该尺度通道数 C（128/256/512），head_dim = C/8；
#   - 逐头在 49 个模板 token 上 softmax（token 数 = token_size² × N 模板）；
#   - W_q/W_k 无偏置（少参数、零初始化友好；记录在报告偏差节）；
#   - W_o：8 头拼接 → 1，即 (1, heads) 线性，无偏置；
#   - γ_ℓ：每尺度一个标量，零初始化 → 训练起点与基线逐位一致（G0 审计）；
#   - **RNG 中立**：全部参数用 torch.empty 建壳 + 专用 torch.Generator 初始化
#     （种子 20261023 + 通道数），构建 CMAR 不消耗全局 RNG → 下游模块初始化
#     与基线逐位一致（G0a/G0b 的前置条件）。
#
# 消融开关（任务书 §3 三格，均由 config 键驱动、默认 = 主设计）：
#   cmar_mode='replace'     → 去掉 γ：corr = attn_readout 直接替换 max 读出（预期负结果）
#   cmar_gamma_init='random'→ γ 随机初始化（专用 generator，N(0, 0.02)，尺度取 0.02
#                             与 localizer.id_null 初始化口径一致，写入报告）
#   cmar_direction='token'  → CMAR-g 字面版：模板 token 为 query、场景位置为 key
#                             （α_{p,t} = softmax_t( (W_q token_t)·(W_k f_p)/√d_h )，
#                             与主设计 q/k 来源互换，softmax 轴不变）
import math

import torch
import torch.nn as nn


class CMARReadout(nn.Module):
    """单个尺度的门控注意力读出（CorrLayer 内部使用，非独立调制器）。"""

    HEADS = 8
    SEED_BASE = 20261023  # 任务书修订日期；+channels 得逐尺度独立种子

    def __init__(self, channels, mode='gate', gamma_init='zero', direction='spatial'):
        super().__init__()
        assert channels % self.HEADS == 0, f'channels={channels} 不能被 {self.HEADS} 头整除'
        assert mode in ('gate', 'replace'), mode
        assert gamma_init in ('zero', 'random'), gamma_init
        assert direction in ('spatial', 'token'), direction
        self.mode = mode
        self.direction = direction
        self.head_dim = channels // self.HEADS
        C, H = channels, self.HEADS
        # torch.empty 不耗全局 RNG；专用 generator 初始化 → 构建不扰动全局随机状态
        g = torch.Generator().manual_seed(self.SEED_BASE + channels)
        self.w_q = nn.Parameter(torch.empty(C, C))
        self.w_k = nn.Parameter(torch.empty(C, C))
        self.w_o = nn.Parameter(torch.empty(1, H))
        nn.init.xavier_uniform_(self.w_q, generator=g)
        nn.init.xavier_uniform_(self.w_k, generator=g)
        nn.init.xavier_uniform_(self.w_o, generator=g)
        # 门控标量 γ：主设计零初始化（训练起点 ≡ 基线）；消融格②随机初始化
        if gamma_init == 'zero':
            self.gamma = nn.Parameter(torch.zeros(1))
        else:
            self.gamma = nn.Parameter(torch.randn(1, generator=g) * 0.02)
        # 训练期 attn_readout 统计（普通 Python 属性，不进 state_dict，不影响 G0a 键集）
        self.reset_log_stats()
        # 探针用：置 True 时缓存注意力权重（仅 probe/可视化开，训练默认关省显存）
        self.keep_attn = False
        self.last_attn = None  # (B, heads, HW, T)，keep_attn=True 时填充

    def reset_log_stats(self):
        self._log_sum = 0.0
        self._log_n = 0
        self._log_min = float('inf')
        self._log_max = float('-inf')

    def pop_log_stats(self):
        """train.py 每 epoch 末取走并清零；无样本时返回 None。"""
        if self._log_n == 0:
            return None
        st = {'mean': self._log_sum / self._log_n,
              'min': self._log_min, 'max': self._log_max, 'n': self._log_n}
        self.reset_log_stats()
        return st

    def forward(self, sim, ir_feat, tok):
        """
        sim:     (B, HW, T)  完整 token 相似度张量（读出前的逐位置×逐 token 余弦相似度）
        ir_feat: (B, C, H, W)  该尺度 IR 场景特征（f_p 来源；HW = H*W）
        tok:     (B, D, T)   相关计算实际使用的模板 token（proj+normalize 之后；D == C）
        return:  (B, HW)     attn_readout
        """
        B, HW, T = sim.shape
        h, d = self.HEADS, self.head_dim
        f = ir_feat.flatten(2).transpose(1, 2)                    # (B, HW, C)
        t = tok.transpose(1, 2)                                   # (B, T, C)
        if self.direction == 'spatial':
            # 主设计：场景位置为 query，模板 token 为 key
            q = (f @ self.w_q.t()).view(B, HW, h, d).transpose(1, 2)   # (B, h, HW, d)
            k = (t @ self.w_k.t()).view(B, T, h, d).transpose(1, 2)    # (B, h, T, d)
            logits = (q @ k.transpose(-1, -2)) / math.sqrt(d)          # (B, h, HW, T)
        else:
            # 消融格③ CMAR-g 字面版：模板 token 为 query，场景位置为 key，
            # 再转置回 (p, t) 取向（softmax 轴仍在 token 维，读出定义不变）
            qt = (t @ self.w_q.t()).view(B, T, h, d).transpose(1, 2)   # (B, h, T, d)
            kp = (f @ self.w_k.t()).view(B, HW, h, d).transpose(1, 2)  # (B, h, HW, d)
            logits = (kp @ qt.transpose(-1, -2)) / math.sqrt(d)        # (B, h, HW, T)
        alpha = logits.softmax(dim=-1)                                 # (B, h, HW, T)
        if self.keep_attn:
            self.last_attn = alpha.detach()
        r = (alpha * sim.unsqueeze(1)).sum(dim=-1)                     # (B, h, HW)
        out = (r.transpose(1, 2) @ self.w_o.t()).squeeze(-1)           # (B, HW)
        if self.training:
            d_out = out.detach()
            self._log_sum += float(d_out.sum())
            self._log_n += d_out.numel()
            self._log_min = min(self._log_min, float(d_out.min()))
            self._log_max = max(self._log_max, float(d_out.max()))
        return out