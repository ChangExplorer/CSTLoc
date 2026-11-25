# IR 搜索图编码器：InfMAE 红外预训练主干（冻结）+ 投影金字塔（实验 F）
#
# 对实验 B（dinov2_corr）单变量：只换 IR 编码器（UNet → InfMAE），其余不动。
# 对比损失关闭（contrast_weight=0）。
#
# 金字塔契约（对齐 UNetEncoder 输出 x1..x5 = 32/64/128/256/512 @ stride 1/2/4/8/16，
# 384×640 输入下 x5 网格 24×40 = InfMAE stride16 输出）：
#   stage3（blocks3+norm，768 @ stride16，24×40）→ proj512 → x5 (512)
#   stage2（blocks2 后，384 @ stride8，48×80）  → proj256 → x4 (256)
#   stage1（blocks1 后，256 @ stride4，96×160） → proj128 → x3 (128)
#   x2 (64 @ stride2)：stage1 投影 64 + upsample×2
#   x1 (32 @ stride1)：轻量 conv stem 吃原图
#
# 权重：/home/user/Code/CSTLoc/infmae/InfMAE.pth 的 'model' 键（243 张量）；
#   加载 strict=False，missing 应只含 decoder_*/mask_token（本编码器只用 encoder 部分）。
#   加载后 requires_grad_(False) + 前向包 torch.no_grad（InfMAE 不参与训练）。
#
# 结构依据：vendor 官方代码 code/models/encoders/infmae_vendor/
#   （models_infmae_skip4.py + vision_transformer.py + pos_embed.py，仅修相对导入与 np.float 兼容）
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

from .infmae_vendor.models_infmae_skip4 import infmae_vit_base_patch16

INFMAE_CKPT = os.environ.get('CSTLOC_INFMAE_CKPT', str(Path(__file__).resolve().parents[3] / 'infmae' / 'InfMAE.pth'))


class _ConvStem(nn.Module):
    """x1 (32 @ stride1)：轻量 conv stem 吃原图。"""
    def __init__(self, in_ch=3, out_ch=32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class IRInfMAEEncoder(nn.Module):
    def __init__(self, img_ch=3, unfreeze_tail=0):
        super().__init__()
        self.unfreeze_tail = int(unfreeze_tail)
        # 官方结构构建（infmae_vit_base_patch16：img_size=[224,56,28], patch_size=[4,2,2],
        # embed_dim=[256,384,768], depth=[2,2,11], num_heads=12, mlp_ratio=[4,4,4]）
        self.bb = infmae_vit_base_patch16()

        # 加载权重（只取 encoder 用到的子模块；missing 应只含 decoder_*/mask_token）
        ck = torch.load(INFMAE_CKPT, map_location='cpu')
        state = ck['model'] if isinstance(ck, dict) and 'model' in ck else ck
        missing, unexpected = self.bb.load_state_dict(state, strict=False)
        self.load_report = {'loaded': len(state),
                            'missing': len(missing),
                            'unexpected': len(unexpected),
                            'missing_prefix': sorted(set(k.split('.')[0] for k in missing))[:6]}
        # 冻结 InfMAE 主干
        self.bb.requires_grad_(False)
        self.bb.eval()
        # 实验 H：解冻末段（blocks3 后 unfreeze_tail 个 block + 最终 norm；默认 0=全冻结不变）
        self.unfrozen_params = []
        if self.unfreeze_tail > 0:
            self.apply_unfreeze(self.unfreeze_tail)
        self._n_frozen = len(self.bb.blocks3) - self.unfreeze_tail

        # 可训练投影金字塔
        self.proj5 = nn.Conv2d(768, 512, 1)     # stage3 → x5 (512 @ stride16)
        self.proj4 = nn.Conv2d(384, 256, 1)     # stage2 → x4 (256 @ stride8)
        self.proj3 = nn.Conv2d(256, 128, 1)     # stage1 → x3 (128 @ stride4)
        self.proj2 = nn.Conv2d(256, 64, 1)      # stage1 → x2 (64 @ stride2)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.stem = _ConvStem(img_ch, 32)       # 原图 → x1 (32 @ stride1)

    def _interp_pos_embed(self, H, W):
        """pos_embed (1,196,768) @14×14 → (1,H/16*W/16,768)，bicubic 插值（固定输入尺寸可缓存）。"""
        pe = self.bb.pos_embed                          # (1,196,768)
        _, N, C = pe.shape
        orig = int(N ** 0.5)                            # 14
        pe2d = pe.reshape(1, orig, orig, C).permute(0, 3, 1, 2)
        pe2d = F.interpolate(pe2d, size=(H // 16, W // 16), mode='bicubic', align_corners=False)
        return pe2d.flatten(2).permute(0, 2, 1)         # (1, L, 768)

    def apply_unfreeze(self, n):
        """解冻 blocks3 末 n 个 block + 最终 norm（实验 H；可重复调用，幂等）。"""
        assert n <= len(self.bb.blocks3), f'unfreeze_tail {n} > blocks3 {len(self.bb.blocks3)}'
        self.unfreeze_tail = int(n)
        for blk in self.bb.blocks3[len(self.bb.blocks3) - n:]:
            for p in blk.parameters():
                p.requires_grad_(True)
        for p in self.bb.norm.parameters():
            p.requires_grad_(True)
        self.unfrozen_params = [p for blk in self.bb.blocks3[len(self.bb.blocks3) - n:]
                                for p in blk.parameters()] + list(self.bb.norm.parameters())
        self._n_frozen = len(self.bb.blocks3) - n

    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W) IR 搜索图（384×640）
        Returns:
            x1..x5 元组（对齐 UNetEncoder 契约，32/64/128/256/512 @ stride 1/2/4/8/16）
        """
        B, _, H, W = x.shape
        # 适配任意输入尺寸：更新 patch_embed 的 img_size 断言（不影响权重/逻辑）
        self.bb.patch_embed1.img_size = (H, W)
        self.bb.patch_embed2.img_size = (H // 4, W // 4)
        self.bb.patch_embed3.img_size = (H // 8, W // 8)
        # 冻结前缀（patch embeds + blocks1/2 + blocks3 前段）包 no_grad 省显存；
        # 解冻尾段（blocks3 后 unfreeze_tail 个 + norm）正常反传（unfreeze_tail=0 时路径与原来一致）
        n_frozen = self._n_frozen
        with torch.no_grad():
            # stage1：patch_embed1 + blocks1（CBlock，无 mask）→ (B,256,H/4,W/4)
            s1 = self.bb.patch_embed1(x)
            for blk in self.bb.blocks1:
                s1 = blk(s1, None)
            # stage2：patch_embed2 + blocks2 → (B,384,H/8,W/8)
            s2 = self.bb.patch_embed2(s1)
            for blk in self.bb.blocks2:
                s2 = blk(s2, None)
            # stage3：patch_embed3 + patch_embed4 + pos_embed(插值) + blocks3 前段 → (B,L,768)
            s3 = self.bb.patch_embed3(s2)
            s3 = s3.flatten(2).permute(0, 2, 1)
            s3 = self.bb.patch_embed4(s3)
            s3 = s3 + self._interp_pos_embed(H, W)
            for blk in self.bb.blocks3[:n_frozen]:
                s3 = blk(s3)
            if self.unfreeze_tail == 0:
                s3 = self.bb.norm(s3)            # 全冻结路径：norm 亦在 no_grad 内（行为不变）
        if self.unfreeze_tail > 0:
            for blk in self.bb.blocks3[n_frozen:]:
                s3 = blk(s3)                     # 解冻段正常反传
            s3 = self.bb.norm(s3)
        h3, w3 = H // 16, W // 16
        s3 = s3.permute(0, 2, 1).reshape(B, 768, h3, w3)

        x5 = self.proj5(s3)                    # (B,512,H/16,W/16)
        x4 = self.proj4(s2)                    # (B,256,H/8,W/8)
        x3 = self.proj3(s1)                    # (B,128,H/4,W/4)
        x2 = self.up2(self.proj2(s1))          # (B,64,H/2,W/2)
        x1 = self.stem(x)                      # (B,32,H,W)
        return x1, x2, x3, x4, x5
