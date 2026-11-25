# 模板编码器：ResNet18 多层特征提取
import torch
import torch.nn as nn
import timm

# 主机上的本地预训练权重路径（默认值，可被 config 的 pretrained_path 覆盖）
DEFAULT_PRETRAINED_PATH = r"/home/user/.cache/huggingface/hub/models--timm--resnet18.fb_swsl_ig1b_ft_in1k/pytorch_model.bin"


class ResNetTemplateEncoder(nn.Module):
    """ResNet18 提取模板切片的多层级特征。

    输出两种表征，供不同调制器消费：
      - vecs:  (s3, s4, s5) 各层 GAP 全局向量 [128, 256, 512]  → FiLM 用
      - maps:  (f3, f4, f5) 各层空间特征图                      → Correlation 用

    多模板：输入 (B, N, 3, 224, 224)，内部 flatten 到 (B*N,...) 编码后 reshape 回来。
    维度对齐 UNet：layer2→x3(128), layer3→x4(256), layer4→x5(512)。

    Args:
        pretrained:      是否加载预训练权重
        pretrained_path: 本地权重文件路径（None 时用 timm 默认下载/缓存）
    """
    def __init__(self, pretrained=True, pretrained_path=DEFAULT_PRETRAINED_PATH):
        super().__init__()
        overlay = None
        if pretrained and pretrained_path:
            overlay = {'file': pretrained_path}
        self.backbone = timm.create_model(
            'resnet18.fb_swsl_ig1b_ft_in1k',
            pretrained=pretrained,
            features_only=True,
            out_indices=(2, 3, 4),
            pretrained_cfg_overlay=overlay,
        )
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, tmpl):
        """
        Args:
            tmpl: (B, N, 3, 224, 224)  N=模板数量
        Returns:
            vecs: tuple of (B, N, C) for C in [128, 256, 512]
            maps: tuple of (B, N, C, h, w)
        """
        B, N = tmpl.shape[:2]
        x = tmpl.flatten(0, 1)                       # (B*N, 3, 224, 224)
        f3, f4, f5 = self.backbone(x)                # 128/256/512 通道
        maps, vecs = [], []
        for f in (f3, f4, f5):
            _, C, h, w = f.shape
            maps.append(f.view(B, N, C, h, w))
            vecs.append(self.gap(f).flatten(1).view(B, N, C))
        return tuple(vecs), tuple(maps)
