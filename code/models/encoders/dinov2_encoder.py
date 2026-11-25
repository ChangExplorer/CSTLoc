# 模板编码器：DINOv2（ViT-L）多层特征提取（实验 B：表征升级）
#
# 与 ResNetTemplateEncoder 同契约（resnet_encoder.py）：
#   输入 (B,N,3,224,224) → 内部 flatten (B*N,3,224,224) → 编码 → reshape 回 (B,N,...)
#   输出 vecs = tuple of (B,N,C)   （各层 GAP 全局向量，FiLM 用）
#         maps = tuple of (B,N,C,h,w)（各层空间特征图，Correlation 用）
#   C = [128, 256, 512]（可训练 1×1 conv 投影对齐 UNet x3/x4/x5）
#
# 归因设计（单变量：只换编码器）：
#   - DINOv2 主干冻结（requires_grad_(False)），前向包 torch.no_grad()
#   - 仅 1×1 conv 投影（1024→128/256/512）与下游调制器/UNet 可训练
#   - 取中间层 n=[7,15,23]（0-indexed block 索引，浅/中/深），reshape=True → (B*N,1024,16,16)
#
# Args:
#   pretrained:      是否加载预训练权重（默认走 timm 缓存，HF_HUB_OFFLINE=1）
#   pretrained_path: 本地权重文件路径（None 时用 timm 缓存/下载）
import torch
import torch.nn as nn
import timm


class DINOv2TemplateEncoder(nn.Module):
    def __init__(self, pretrained=True, pretrained_path=None):
        super().__init__()
        overlay = None
        if pretrained and pretrained_path:
            overlay = {'file': pretrained_path}
        self.backbone = timm.create_model(
            'vit_large_patch14_dinov2.lvd142m',
            pretrained=pretrained,
            img_size=224,
            pretrained_cfg_overlay=overlay,
        )
        # 冻结 DINOv2 主干：不参与训练
        self.backbone.requires_grad_(False)
        self.backbone.eval()

        # 可训练 1×1 投影：1024 → [128, 256, 512]（对齐 UNet x3/x4/x5）
        self.proj128 = nn.Conv2d(1024, 128, 1)
        self.proj256 = nn.Conv2d(1024, 256, 1)
        self.proj512 = nn.Conv2d(1024, 512, 1)
        self.gap = nn.AdaptiveAvgPool2d(1)

    def intermediate_taps(self, x, taps=(15, 23)):
        """实验 J：暴露冻结 DINOv2 中间层特征（reshape=True）供身份专家 head 使用。
        现有 forward 契约不变。返回 [(B,1024,h,w), ...]。"""
        with torch.no_grad():
            return self.backbone.get_intermediate_layers(x, n=list(taps), reshape=True)

    def intermediate_gap(self, x, taps=(15, 23)):
        """实验 J2：GAP 口径与 tools/preextract_ucm_feats.py 第 40-43 行逐式一致——
        reshape=False 取中间层（含 CLS），每 tap 对全部 token（含 CLS，257 个）做 mean(dim=1)，
        两 tap concat → (B,2048)。注意：intermediate_taps 的 reshape=True 是 256 patch token，
        口径不同，不得复用。"""
        with torch.no_grad():
            outs = self.backbone.get_intermediate_layers(x, n=list(taps), reshape=False)
            return torch.cat([o.mean(dim=1) for o in outs], dim=1)

    def forward(self, tmpl):
        """
        Args:
            tmpl: (B, N, 3, 224, 224)
        Returns:
            vecs: tuple of (B, N, C) for C in [128, 256, 512]
            maps: tuple of (B, N, C, h, w)
        """
        B, N = tmpl.shape[:2]
        x = tmpl.flatten(0, 1)                       # (B*N, 3, 224, 224)
        with torch.no_grad():
            feats = self.backbone.get_intermediate_layers(
                x, n=[7, 15, 23], reshape=True)      # (B*N,1024,16,16) × 3
        maps, vecs = [], []
        for f, proj in zip(feats, (self.proj128, self.proj256, self.proj512)):
            p = proj(f)                              # (B*N, C, 16, 16)
            _, C, h, w = p.shape
            maps.append(p.view(B, N, C, h, w))
            vecs.append(self.gap(p).flatten(1).view(B, N, C))
        return tuple(vecs), tuple(maps)
