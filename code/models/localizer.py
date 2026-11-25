# 组装器：encoder + modulator + UNet 编解码，按 config 选型
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

# 实验 J3：常数注入模式的固定向量尺度 s = head_v0 在 val 模板嵌入上的平均 L2 范数
# （1355 张 val 模板切片嵌入，diag_out/j3_scale_stat.log 实测 1.9098；训练 n_templates=1
# 时 id_emb 即单切片嵌入，故取"全部切片"口径。生成口径：torch.Generator().manual_seed(0)
# 的 N(0,1) 256-d 向量，L2 归一化后乘 s。）
J3_NULL_CONSTANT_SCALE = 1.9098

from .unet_backbone import UNetEncoder, UNetDecoder
from .encoders.resnet_encoder import ResNetTemplateEncoder
from .encoders.dinov2_encoder import DINOv2TemplateEncoder
from .encoders.ir_infmae_encoder import IRInfMAEEncoder
from .modulators.film import FiLMModulator
from .modulators.correlation import CorrelationModulator
from .modulators.simple import SimpleFusionModulator

_ENCODERS = {'resnet18': ResNetTemplateEncoder, 'dinov2': DINOv2TemplateEncoder}
_IR_ENCODERS = {'unet': UNetEncoder, 'infmae': IRInfMAEEncoder}
_MODULATORS = {'film': FiLMModulator, 'correlation': CorrelationModulator, 'simple': SimpleFusionModulator}


class Localizer(nn.Module):
    """VIS 模板引导的 IR 目标定位网络。

    数据流：
      模板 (B,N,3,224,224) → encoder → (vecs, maps)
      IR   (B,3,H,W)       → UNet encoder → x1..x5
      (x3,x4,x5) 经 modulator 用模板表征调制
      → UNet decoder → 定位热图 logits (B,1,H,W)
    """
    def __init__(self, encoder='resnet18', modulator='correlation',
                 pretrained=True, pretrained_path=None, img_ch=3, ir_encoder=None,
                 use_id=False, identity_expert=None, mask_null_p=0.0, id_dim=256,
                 cmar=False, cmar_mode='gate', cmar_gamma_init='zero',
                 cmar_direction='spatial'):
        super().__init__()
        # IR 编码器选型：参数优先；否则读环境变量 CSTLOC_IR_ENCODER（实验 F 用，
        # train.py 为判卷标准只读不可改，故以环境变量传递选型）；默认 unet
        if ir_encoder is None:
            ir_encoder = os.environ.get('CSTLOC_IR_ENCODER', 'unet')
        enc_kwargs = {'pretrained': pretrained}
        if pretrained_path is not None:
            enc_kwargs['pretrained_path'] = pretrained_path
        self.tmpl_encoder = _ENCODERS[encoder](**enc_kwargs)
        # 任务书 2.5：CMAR 键透传（仅 correlation 调制器接受；默认 cmar=False 逐位不变）
        if cmar:
            assert modulator == 'correlation', f'CMAR 仅适配 correlation 调制器，实际 {modulator}'
        mod_kw = (dict(use_id=use_id, id_dim=id_dim) if use_id else {})
        if modulator == 'correlation':
            mod_kw.update(cmar=cmar, cmar_mode=cmar_mode,
                          cmar_gamma_init=cmar_gamma_init, cmar_direction=cmar_direction)
        self.modulator = _MODULATORS[modulator](**mod_kw)
        self.ir_encoder = _IR_ENCODERS[ir_encoder](img_ch=img_ch)
        self.decoder = UNetDecoder(output_ch=1)
        # 实验 J：身份专家（冻结）+ null token
        self.use_id = use_id
        self.id_dim = id_dim
        self.mask_null_p = mask_null_p
        self.expert = None
        # 实验 J3：identity_expert='null_constant' → 常数注入模式（无专家头模块，
        # fc/BNNeck/cls 均不构建），id_emb = 冻结常数向量，量级与真实专家同量级
        self.null_constant = (identity_expert == 'null_constant')
        if identity_expert and not self.null_constant:
            from .identity_expert import IdentityExpert
            self.expert = IdentityExpert(identity_expert)
        if use_id:
            self.id_null = nn.Parameter(torch.randn(id_dim) * 0.02)
            if self.null_constant:
                g = torch.Generator().manual_seed(0)
                v = torch.randn(id_dim, generator=g)
                v = v / v.norm()
                self.register_buffer('null_const', v * J3_NULL_CONSTANT_SCALE)

    def _compute_id_emb(self, templates, mask_null=False):
        """模板 N 切片各过冻结专家得 256-d，取 mean → (B,256)；mask_null 时按 p 替换为 null。
        返回 (id_emb (B,256), id_mask (B,) bool)——mask=True 表示该样本被遮蔽。
        J2：linear 系专家走 intermediate_gap（(B*N,2048) GAP 口径，与预提取缓存逐式一致）；
        spatial 专家走 intermediate_taps（reshape=True，J 口径）。
        J3：null_constant 模式 → id_emb = 冻结常数向量（与模板输入无关）。"""
        B, N = templates.shape[:2]
        if self.null_constant:
            id_emb = self.null_const.unsqueeze(0).expand(B, -1).clone()
        else:
            x = templates.flatten(0, 1)
            if self.expert.arch.startswith('linear'):
                feats = self.tmpl_encoder.intermediate_gap(x)   # (B*N,2048)
                ids = self.expert(feats)                        # (B*N,256)
            else:
                maps_list = [t for t in self.tmpl_encoder.intermediate_taps(x)]
                ids = self.expert(maps_list)                    # (B*N,256)
            id_emb = ids.view(B, N, self.id_dim).mean(dim=1)    # (B,256)
        id_mask = torch.zeros(B, dtype=torch.bool, device=id_emb.device)
        if mask_null and self.mask_null_p > 0:
            id_mask = torch.rand(B, device=id_emb.device) < self.mask_null_p
            id_emb = id_emb.clone()
            id_emb[id_mask] = self.id_null
        return id_emb, id_mask

    def forward(self, ir_image, templates, return_aux=False, ir_feats=None, mask_null=False):
        """
        ir_image:  (B, 3, H, W)
        templates: (B, N, 3, 224, 224)
        return_aux=True 时返回 (logits, {'x5': 调制后 x5, 'proto': 模板原型 (B,512),
        'id_emb': 身份嵌入 (B,256), 'id_mask': 遮蔽掩码 (B,)})，供训练态使用；默认仅返回 logits。
        ir_feats:  传入时跳过 IR 编码，直接用缓存特征 (x1..x5)（对偶模板监督共享 IR 特征用）。
        mask_null: 训练态模板随机遮蔽（use_id 时生效，按 mask_null_p 替换 id_emb 为 null）。
        """
        vecs, maps = self.tmpl_encoder(templates)
        if ir_feats is None:
            x1, x2, x3, x4, x5 = self.ir_encoder(ir_image)
        else:
            x1, x2, x3, x4, x5 = ir_feats
        id_emb = id_mask = None
        if self.use_id and (self.expert is not None or self.null_constant):
            id_emb, id_mask = self._compute_id_emb(templates, mask_null=mask_null)
        x3, x4, x5 = self.modulator((x3, x4, x5), vecs, maps, id_emb=id_emb)
        logits = self.decoder(x1, x2, x3, x4, x5)
        if return_aux:
            proto = vecs[2].mean(dim=1)          # (B,512) 模板原型（512 层 N 平均）
            return logits, {'x5': x5, 'proto': proto, 'id_emb': id_emb, 'id_mask': id_mask}
        return logits
