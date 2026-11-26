# UNet 主干组件：从原 model.py 抽出，供 localizer 组装
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class Up(nn.Module):
    """上采样 + 与 skip 拼接 + ConvBlock。

    in_ch: 来自更深层的通道数；out_ch: skip 通道数（也是输出通道数）。
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = ConvBlock(in_ch + out_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        return self.conv(torch.cat([x2, x1], dim=1))


class UNetEncoder(nn.Module):
    """IR 场景图编码器，输出 5 层特征 x1..x5（通道 32/64/128/256/512）。"""
    def __init__(self, img_ch=3):
        super().__init__()
        self.Maxpool = nn.MaxPool2d(2, 2)
        self.Conv1 = ConvBlock(img_ch, 32)
        self.Conv2 = ConvBlock(32, 64)
        self.Conv3 = ConvBlock(64, 128)
        self.Conv4 = ConvBlock(128, 256)
        self.Conv5 = ConvBlock(256, 512)

    def forward(self, x):
        x1 = self.Conv1(x)
        x2 = self.Conv2(self.Maxpool(x1))
        x3 = self.Conv3(self.Maxpool(x2))
        x4 = self.Conv4(self.Maxpool(x3))
        x5 = self.Conv5(self.Maxpool(x4))
        return x1, x2, x3, x4, x5


class UNetDecoder(nn.Module):
    """解码器 + 1x1 输出头，输出定位 logits (B,1,H,W)。

    输出头 bias 初始化为 -4.0：sigmoid(-4)≈0.018，训练初期输出接近全背景，
    与稀疏高斯热图匹配，稳定收敛。
    """
    def __init__(self, output_ch=1):
        super().__init__()
        self.Up5 = Up(512, 256)
        self.Up4 = Up(256, 128)
        self.Up3 = Up(128, 64)
        self.Up2 = Up(64, 32)
        self.Conv_1x1 = nn.Conv2d(32, output_ch, 1)
        nn.init.constant_(self.Conv_1x1.bias, -4.0)

    def forward(self, x1, x2, x3, x4, x5):
        d5 = self.Up5(x5, x4)
        d4 = self.Up4(d5, x3)
        d3 = self.Up3(d4, x2)
        d2 = self.Up2(d3, x1)
        return self.Conv_1x1(d2)
