# 调制器抽象接口：统一契约，供 localizer 无差别调用
#
# 契约：modulate(ir_feats, tmpl_vecs, tmpl_maps) -> 调制后的 (x3, x4, x5)
#   ir_feats:  (x3, x4, x5) UNet 编码器的三层特征，通道 [128, 256, 512]
#   tmpl_vecs: (B, N, C) 三元组，模板全局向量（FiLM 消费）
#   tmpl_maps: (B, N, C, h, w) 三元组，模板空间特征图（Correlation 消费）
#
# 每个调制器只用它需要的那一路输入，另一路忽略。
# 三层对应 UNet 的 x3(128)/x4(256)/x5(512=bottleneck)，即 bottleneck + 最下两层 skip。
from abc import ABC, abstractmethod
import torch.nn as nn


class BaseModulator(nn.Module, ABC):
    # 该调制器需要哪种模板表征：'vec'（全局向量）或 'map'（空间特征图）
    needs: str = None

    @abstractmethod
    def forward(self, ir_feats, tmpl_vecs, tmpl_maps):
        """返回调制后的 (x3, x4, x5)。"""
        raise NotImplementedError
