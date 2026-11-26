# 评测：PCK@0.5 + 按难度分层 + 干扰物诊断
#
# PCK@0.5：预测峰值中心落在 GT 中心 0.5·max(w,h) 内即命中。
# 分层：overall / easy / hard，回应难度分级设计。
import torch
import torch.nn.functional as F


def peak_center(logits):
    """热图峰值坐标 (B,2) = (cx, cy)（像素）。"""
    B, _, H, W = logits.shape
    flat = logits.view(B, -1)
    idx = flat.argmax(dim=1)
    cy = (idx // W).float()
    cx = (idx % W).float()
    return torch.stack([cx, cy], dim=1)


@torch.no_grad()
def evaluate(model, loader, device, pck_thresh=0.5):
    model.eval()
    buckets = {'overall': [0, 0], 'easy': [0, 0], 'hard': [0, 0]}

    for batch in loader:
        ir = batch['ir'].to(device)
        templates = batch['templates'].to(device)
        gt = batch['gt_center'].to(device)           # (B,2)
        size = batch['obj_size'].to(device)          # (B,2) w,h

        logits = model(ir, templates)
        pred = peak_center(logits).to(device)

        dist = torch.norm(pred - gt, dim=1)
        thresh = pck_thresh * torch.maximum(size[:, 0], size[:, 1])
        hit = (dist <= thresh)

        for i in range(ir.size(0)):
            h = int(hit[i].item())
            buckets['overall'][0] += h
            buckets['overall'][1] += 1
            diff = batch['difficulty'][i]
            if diff in ('easy', 'hard'):
                buckets[diff][0] += h
                buckets[diff][1] += 1

    return {k: (c / n if n else 0.0, n) for k, (c, n) in buckets.items()}


def format_results(res):
    lines = []
    for k in ('overall', 'easy', 'hard'):
        acc, n = res[k]
        lines.append(f"{k:8s} PCK@0.5 = {acc:.4f}  (n={n})")
    return '\n'.join(lines)
