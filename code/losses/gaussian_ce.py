# 任务书 1.9：Gaussian CrossEntropy 读出监督（MARCO util/losses.py::GaussianCrossEntropyLoss
# 思想移植——references/repos/MARCO，Apache-2.0；逐函数重写、零 import 耦合）。
#
# 关键写死差异（任务书陷阱节）：MARCO 的 sim 是 DINOv2 未归一化点积、softmax 温度 β=10
# （除法）；我们的 corr 是余弦 ∈[−1,1]，直搬 τ=10 会退化为均匀分布 → 写死 τ=0.07
# （余弦标定标准值，G0d sanity 核验）。
#
# 目标构造（train.py 侧）：x4 高斯目标 = avg-pool 全分辨率数据集 heatmap 至 x4 格 +
# σ(t) 余弦退火（1.5→0.5，kernel 9）高斯卷积 + 逐样本归一；σ 单位 = x4 特征格
# （MARCO σ 3.5@180² 等比缩放 ≈ 0.93@48×80，UAV 小目标取 1.5→0.5）。
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def sigma_cosine_anneal(step, total_steps, sigma0=1.5, sigma1=0.5):
    """σ 余弦退火 1.5→0.5（随训练全程，MARCO WarmupCosine 的退火思想；σ 单位=特征格）。"""
    if total_steps <= 0:
        return sigma0
    t = min(step / total_steps, 1.0)
    return sigma1 + (sigma0 - sigma1) * 0.5 * (1.0 + math.cos(math.pi * t))


def gaussian_kernel_2d(size=9, sigma=1.0):
    """归一化截断 2D 高斯核（size×size，σ 单位=特征格）。"""
    ax = torch.arange(size, dtype=torch.float32) - (size - 1) / 2
    g = torch.exp(-(ax ** 2) / (2 * sigma ** 2))
    k = g[:, None] * g[None, :]
    return (k / k.sum()).view(1, 1, size, size)


def build_ce_target(heatmap, sigma, kernel_size=9):
    """x4 读出监督 CE 目标：(B,1,H,W) 全分辨率数据集 heatmap → avg-pool 至 corr 图尺寸 →
    σ 高斯模糊（kernel_size，padding=kernel_size//2）→ 逐样本归一 sum=1。
    返回 (B,1,Hl,Wl)。"""
    B = heatmap.shape[0]
    hl, wl = heatmap.shape[2] // 8, heatmap.shape[3] // 8   # x4 = stride 8（InfMAE UNet）
    t = F.adaptive_avg_pool2d(heatmap, (hl, wl))
    k = gaussian_kernel_2d(kernel_size, sigma).to(t.device)
    t = F.conv2d(t, k, padding=kernel_size // 2)
    flat = t.reshape(B, -1)
    flat = flat / flat.sum(dim=1, keepdim=True).clamp(min=1e-6)
    return flat.reshape_as(t)


class GaussianCrossEntropyLoss(nn.Module):
    """softmax(τ) over 空间 × 高斯目标 CE。corr: (B,1,Hl,Wl)；target: (B,1,Hl,Wl) 逐样本
    归一（build_ce_target 产物）。τ=0.07 写死（余弦标定）。"""

    def __init__(self, tau=0.07):
        super().__init__()
        self.tau = tau

    def forward(self, corr, target):
        B = corr.shape[0]
        logits = corr.reshape(B, -1) / self.tau
        logp = F.log_softmax(logits, dim=1)
        t = target.reshape(B, -1)
        return -(t * logp).sum(dim=1).mean()
