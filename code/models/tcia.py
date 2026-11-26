# TCIA独立验证器：共享冻结DINOv2 + 模板/候选两套手动rank-4 LoRA + 共享256-d AttnPool。
# 不依赖peft，不修改timm/Localizer。LoRA disabled时每个包装层严格只走原Linear，raw与现役2048-d余弦同式。
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class DualLoRALinear(nn.Module):
    """原Linear冻结；A/B各角色独立。enabled=False严格原路，enabled=True加B@A残差。"""
    def __init__(self, base, rank=4, alpha=4):
        super().__init__()
        self.base = base
        self.base.requires_grad_(False)
        self.rank, self.alpha = rank, alpha
        self.role, self.enabled = 'tmpl', False
        for role in ('tmpl', 'cand'):
            a = nn.Parameter(torch.empty(rank, base.in_features))
            b = nn.Parameter(torch.zeros(base.out_features, rank))
            nn.init.kaiming_uniform_(a, a=math.sqrt(5))
            self.register_parameter(f'lora_{role}_a', a)
            self.register_parameter(f'lora_{role}_b', b)

    def forward(self, x):
        if not self.enabled:
            return self.base(x)
        a = getattr(self, f'lora_{self.role}_a')
        b = getattr(self, f'lora_{self.role}_b')
        return self.base(x) + F.linear(F.linear(x, a), b) * (self.alpha / self.rank)


class AttnPool(nn.Module):
    """共享单查询AttnPool：256个patch token→256-d归一化身份向量。"""
    def __init__(self, d=1024, out=256):
        super().__init__()
        self.query = nn.Parameter(torch.randn(d) * 0.02)
        self.proj = nn.Linear(d, out)

    def forward(self, tok):
        w = torch.softmax(torch.einsum('bnd,d->bn', tok, self.query) / math.sqrt(tok.shape[-1]), dim=1)
        z = torch.einsum('bn,bnd->bd', w, tok)
        return F.normalize(self.proj(z), dim=-1)


class TCIA(nn.Module):
    def __init__(self, rank=4, alpha=4, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model('vit_large_patch14_dinov2.lvd142m',
                                         pretrained=pretrained, img_size=224)
        self.backbone.requires_grad_(False)
        self.backbone.eval()
        self.wrapped = []
        for block in self.backbone.blocks:
            for mod in (block.attn.qkv, block.mlp.fc1, block.mlp.fc2):
                assert isinstance(mod, nn.Linear), type(mod)
            block.attn.qkv = DualLoRALinear(block.attn.qkv, rank, alpha)
            block.mlp.fc1 = DualLoRALinear(block.mlp.fc1, rank, alpha)
            block.mlp.fc2 = DualLoRALinear(block.mlp.fc2, rank, alpha)
            self.wrapped.extend([block.attn.qkv, block.mlp.fc1, block.mlp.fc2])
        self.head = AttnPool(1024, 256)

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()  # 冻结主干始终eval
        return self

    def set_lora(self, role, enabled=True):
        assert role in ('tmpl', 'cand')
        for m in self.wrapped:
            m.role, m.enabled = role, enabled

    def encode_raw(self, x):
        self.set_lora('tmpl', False)  # 所有wrapped disabled；role不影响
        out = self.backbone.get_intermediate_layers(x, n=[15, 23], reshape=False)
        return torch.cat([t.mean(dim=1) for t in out], dim=1)

    def encode_head(self, x, role):
        self.set_lora(role, True)
        tok = self.backbone.forward_features(x)  # 24个Block+最终norm，确保最后层LoRA也在损失路径
        return self.head(tok)

    def scores(self, tmpl_x, cand_x, lora_enabled=True):
        if not lora_enabled:
            zt = F.normalize(self.encode_raw(tmpl_x), dim=-1)
            zc = F.normalize(self.encode_raw(cand_x), dim=-1)
        else:
            zt = self.encode_head(tmpl_x, 'tmpl')
            zc = self.encode_head(cand_x, 'cand')
        return zt, zc

    def adapter_state_dict(self):
        return {k: v.detach().cpu() for k, v in self.state_dict().items()
                if '.lora_' in k or k.startswith('head.')}

    def load_adapter_state_dict(self, state, strict=True):
        own = self.state_dict()
        missing = [k for k in self.adapter_state_dict() if k not in state]
        if strict:
            assert not missing, missing
        own.update(state)
        self.load_state_dict(own, strict=True)

    def trainable_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class TCIAv2(TCIA):
    """Released TCIA（top-20 峰 crop 变体）。结构同 TCIA，MODEL_VERSION 用于版本标识。"""
    MODEL_VERSION = 'tcia_v2_peakcrop_rank4_attn256'
