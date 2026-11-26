# 实验 P-a：目标-干扰物 margin 项（任务书 QP 分支 A 唯一新增损失）
#
# 物理含义（任务书 §P3）：现有抑制项只是无差别压低干扰物区域响应（MSE 式），
# 没有任何项要求「目标响应必须胜过干扰物响应」——margin 项补上这一缺口
# （DaSiamRPN 干扰物感知的热图版）。
#
# L_dist = mean_{i∈有效} relu( 1.0 − ( s_gt,i − max_{d∈干扰物mask_i} s_d,i ) )
#   s = sigmoid(logits)；s_gt,i = 样本 i GT 中心像素的 s 值（与 dual-template 的
#   s_pos[GT中心] 同取法：heatmap argmax 落格）；干扰物区域 = suppression > 0.5；
#   仅对「含 ≥1 干扰物」的样本生效；调用方负责传入 valid 掩码（如 dual-template 的
#   未遮蔽样本掩码）。weight 由 config distractor_margin 控制（分支 A 固定 1.0）。
import torch


def distractor_margin_loss(logits, heatmap, suppression, valid=None):
    """logits/heatmap/suppression: (B,1,H,W)；valid: (B,) bool 或 None（全样本）。
    返回标量 loss（有效样本上的 mean；无有效样本返回 0）。"""
    s = torch.sigmoid(logits)
    gt_idx = heatmap.flatten(1).argmax(dim=1)
    s_gt = s.flatten(1).gather(1, gt_idx.unsqueeze(1)).squeeze(1)
    dis_mask = (suppression > 0.5).flatten(1)
    has_dis = dis_mask.any(dim=1)
    s_dmax = s.flatten(1).masked_fill(~dis_mask, -1e9).max(dim=1).values
    margin = torch.relu(1.0 - (s_gt - s_dmax))
    if valid is not None:
        margin = margin[valid & has_dis]
    else:
        margin = margin[has_dis]
    if margin.numel() == 0:
        return logits.new_zeros(())
    return margin.mean()
