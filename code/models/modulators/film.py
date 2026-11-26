# FiLM 调制器（baseline）：全局通道级仿射调制
#
# 复刻原项目机制：模板全局向量 → gamma/beta → 对 IR 特征做逐通道 affine。
# 多模板：对 N 个模板向量做平均池化（全局调制无法利用空间/多样性，只能平均）。
# 这正是本次实验要对比的"全局调制"，作为 Correlation 的对照。
import torch
import torch.nn as nn

from .base import BaseModulator


class FiLMLayer(nn.Module):
    def __init__(self, semantic_dim, feature_channels):
        super().__init__()
        self.gamma_fc = nn.Linear(semantic_dim, feature_channels)
        self.beta_fc = nn.Linear(semantic_dim, feature_channels)
        # gamma≈1, beta≈0：初始为恒等，不破坏 UNet 已有特征
        nn.init.zeros_(self.gamma_fc.weight)
        nn.init.ones_(self.gamma_fc.bias)
        nn.init.zeros_(self.beta_fc.weight)
        nn.init.zeros_(self.beta_fc.bias)

    def forward(self, feature, semantic_vec):
        gamma = self.gamma_fc(semantic_vec).unsqueeze(-1).unsqueeze(-1)
        beta = self.beta_fc(semantic_vec).unsqueeze(-1).unsqueeze(-1)
        return gamma * feature + beta


class FiLMModulator(BaseModulator):
    needs = 'vec'

    def __init__(self):
        super().__init__()
        # 语义维 = 通道数：x3←layer2(128), x4←layer3(256), x5←layer4(512)
        self.film_x3 = FiLMLayer(128, 128)
        self.film_x4 = FiLMLayer(256, 256)
        self.film_x5 = FiLMLayer(512, 512)

    def forward(self, ir_feats, tmpl_vecs, tmpl_maps):
        x3, x4, x5 = ir_feats
        v3, v4, v5 = (v.mean(dim=1) for v in tmpl_vecs)   # (B,N,C)→(B,C) 多模板平均
        x3 = self.film_x3(x3, v3)
        x4 = self.film_x4(x4, v4)
        x5 = self.film_x5(x5, v5)
        return x3, x4, x5
