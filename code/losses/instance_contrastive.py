# 实例级对比损失（实验 C1）：干扰物=硬负样本，InfoNCE
#
# 设计（领导拍板，MODAL/DSMCL 式硬负样本对比）：
#   - anchor     = 模板原型 proto：模板 vecs 512 层（对应 x5）的 N 平均，(B,512)
#   - 正样本     = GT 中心处调制后 x5 特征（坐标从 heatmap argmax 推出，不改数据集）
#   - 硬负样本   = 抑制掩码 top-K 峰值位置特征（K=8，干扰物中心）
#   - 背景负样本 = 随机 K 点特征（K=8）
#   - 余弦相似度 InfoNCE，温度 0.07
#
# 坐标映射：heatmap/suppression 为 (B,1,H,W) 像素图，x5 为 (B,512,h,w)，
#   stride = H/h = W/w（UNet 5 层 4 次 pool → 384/24 = 640/40 = 16）。
import torch
import torch.nn as nn
import torch.nn.functional as F


class InstanceContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07, k_hard=8, k_bg=8, stride=16, seed=0):
        super().__init__()
        self.temperature = temperature
        self.k_hard = k_hard
        self.k_bg = k_bg
        self.stride = stride
        self.seed = seed

    def _feat_coords(self, idx_pix, H, W, h, w):
        """展平像素索引 (B,K) → 特征图 (row, col) 坐标，clamp 到有效范围。"""
        cy = idx_pix // W
        cx = idx_pix % W
        fy = (cy.float() / self.stride).round().long().clamp(0, h - 1)
        fx = (cx.float() / self.stride).round().long().clamp(0, w - 1)
        return fy, fx

    def forward(self, x5, proto, heatmap, suppression):
        """
        Args:
            x5:          (B,512,h,w) 调制后深层特征（UNet x5）
            proto:       (B,512) 模板原型（模板 vecs 512 层 N 平均）
            heatmap:     (B,1,H,W) GT 高斯热图（argmax=正样本中心）
            suppression: (B,1,H,W) 干扰物抑制图（top-K 峰值=硬负样本位置）
        Returns:
            标量 InfoNCE loss（有限、非 NaN；正样本与硬/背景负样本对 proto 余弦）
        """
        B, C, h, w = x5.shape
        H, W = heatmap.shape[2], heatmap.shape[3]

        # 正样本：GT 中心特征
        gt_idx = heatmap.flatten(1).argmax(dim=1)                 # (B,)
        gy, gx = self._feat_coords(gt_idx, H, W, h, w)
        pos = x5[torch.arange(B, device=x5.device), :, gy, gx]    # (B,512)

        # 硬负样本：抑制掩码 top-K 峰值位置特征
        flat_s = suppression.flatten(1)                           # (B,H*W)
        topk_idx = flat_s.topk(self.k_hard, dim=1).indices        # (B,K)
        hy, hx = self._feat_coords(topk_idx, H, W, h, w)
        b_idx = torch.arange(B, device=x5.device)[:, None].expand(-1, self.k_hard)
        hard = x5[b_idx, :, hy, hx]                               # (B,K,512)

        # 背景负样本：随机 K 点特征（固定种子，确定性）
        gen = torch.Generator(device=x5.device).manual_seed(self.seed)
        bg_idx = torch.randint(0, H * W, (B, self.k_bg), generator=gen, device=x5.device)
        by_, bx_ = self._feat_coords(bg_idx, H, W, h, w)
        b_idx2 = torch.arange(B, device=x5.device)[:, None].expand(-1, self.k_bg)
        bg = x5[b_idx2, :, by_, bx_]                              # (B,K,512)

        # 余弦相似度 InfoNCE
        proto_n = F.normalize(proto, dim=-1)
        pos_n = F.normalize(pos, dim=-1)
        neg_n = F.normalize(torch.cat([hard, bg], dim=1), dim=-1)  # (B,K_h+K_b,512)

        sim_pos = (proto_n * pos_n).sum(-1) / self.temperature                   # (B,)
        sim_neg = (proto_n.unsqueeze(1) * neg_n).sum(-1) / self.temperature     # (B,K_total)
        logits_c = torch.cat([sim_pos.unsqueeze(1), sim_neg], dim=1)
        labels = torch.zeros(B, dtype=torch.long, device=x5.device)
        return F.cross_entropy(logits_c, labels)
