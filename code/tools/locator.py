"""Frozen dual-backbone locator: construction, cached forward, peak extraction.

The released locator is DINOv2 ViT-L (template) + InfMAE ViT-B (IR scene),
three-scale CorrLayer, UNet heatmap head, plus the code-generated random
identity expert (see models/identity_expert.py).
"""
import os
from pathlib import Path

import numpy as np
import torch

# 发布定位器使用正交投影适配器（见 configs/final/locator.yaml: projection_adapter=orth）。
os.environ.setdefault('CSTLOC_PROJ_ADAPTER', 'orth')

from models.localizer import Localizer
from tools.image_ops import K_PEAKS, MIN_DIST, crop_patch, nms_peaks
from tools.paths import weights_dir

DEVICE = 'cuda'


def default_locator_ckpt() -> Path:
    v = os.environ.get('CSTLOC_LOCATOR_CKPT')
    if v:
        return Path(v)
    return weights_dir() / 'locator' / 'best_model.pth'


def build_locator():
    # 默认发布定位器；可经 env 指定不同定位器/模块做消融。
    ckpt = Path(os.environ.get('CSTLOC_LOCATOR_CKPT', str(default_locator_ckpt())))
    if not ckpt.exists():
        raise SystemExit(
            f'locator checkpoint not found: {ckpt}\n'
            'Expected weights/locator/best_model.pth (produced by scripts/02_train_locator.sh) '
            'or set CSTLOC_LOCATOR_CKPT.')
    modulator = os.environ.get('CSTLOC_LOCATOR_MODULATOR', 'correlation')
    use_id = os.environ.get('CSTLOC_LOCATOR_USE_ID', '1') == '1'
    expert = os.environ.get('CSTLOC_LOCATOR_EXPERT', 'random') if use_id else None
    model = Localizer(encoder='dinov2', modulator=modulator, pretrained=True,
                      ir_encoder='infmae', use_id=use_id, identity_expert=expert,
                      mask_null_p=0.0).to(DEVICE)
    model.load_state_dict(torch.load(ckpt, map_location='cpu'), strict=True)
    model.eval()
    return model


def in_box(px, py, b):
    return b[0] <= px <= b[2] and b[1] <= py <= b[3]


def label_peaks(peaks, gt_box, dboxes):
    """2 = inside GT box, 1 = inside a distractor box, 0 = background."""
    labs = []
    for x, y, _v in peaks:
        if in_box(x, y, gt_box):
            labs.append(2)
        elif any(in_box(x, y, b) for b in dboxes):
            labs.append(1)
        else:
            labs.append(0)
    return labs


def forward_cached(model, use_id, ir_x, feats):
    """缓存组合前向（与 Localizer.forward 逐模块同路径，模板塔走缓存原始特征）。
    ir_x: (B_ir,3,384,640) cuda（通常 B_ir=1：单帧多模板）；feats: list 长度 B_tmpl。
    ir_feats expand 到模板批量（与标准 forward 的同 batch 语义一致）。"""
    B = len(feats)
    m7 = torch.stack([d['m7'] for d in feats]).to(DEVICE)
    m15 = torch.stack([d['m15'] for d in feats]).to(DEVICE)
    m23 = torch.stack([d['m23'] for d in feats]).to(DEVICE)
    p7 = model.tmpl_encoder.proj128(m7)
    p15 = model.tmpl_encoder.proj256(m15)
    p23 = model.tmpl_encoder.proj512(m23)
    vecs = tuple(model.tmpl_encoder.gap(p).flatten(1).unsqueeze(1) for p in (p7, p15, p23))
    maps = tuple(p.unsqueeze(1) for p in (p7, p15, p23))
    if use_id:
        if getattr(model, 'null_constant', False):
            id_emb = model.null_const.unsqueeze(0).expand(B, -1).clone()
        else:
            gap = torch.stack([d['gap'] for d in feats]).to(DEVICE)
            id_emb = model.expert(gap)
    else:
        id_emb = None
    x1, x2, x3, x4, x5 = model.ir_encoder(ir_x)
    if B != ir_x.shape[0]:
        x1, x2, x3, x4, x5 = (t.expand(B, -1, -1, -1) for t in (x1, x2, x3, x4, x5))
    x3, x4, x5 = model.modulator((x3, x4, x5), vecs, maps, id_emb=id_emb)
    return model.decoder(x1, x2, x3, x4, x5)


def crops_from_peaks(ir_bgr, peaks):
    """所有train/val/test、缓存复用/新提取均只经此224 crop入口。"""
    return [crop_patch(ir_bgr, int(x), int(y)) for x, y, _v in peaks]


@torch.no_grad()
def extract_peaks_and_crops(model, tmpl_entries, fr, ir_bgr, use_id=None):
    """train/val/test唯一现场取峰入口：固定B≤12。
    输入同帧模板列表，返回逐trial (20峰, 224 crops)；候选crop只经crops_from_peaks。
    use_id=None 时按模型自身 use_id 选择。"""
    assert 1 <= len(tmpl_entries) <= 12
    if use_id is None:
        use_id = bool(getattr(model, 'use_id', True))
    logits = forward_cached(model, use_id, fr['ir'].unsqueeze(0).to(DEVICE),
                            list(tmpl_entries))
    heats = torch.sigmoid(logits)[:, 0].cpu().numpy()
    peaks = [nms_peaks(h, K_PEAKS, MIN_DIST) for h in heats]
    return peaks, [crops_from_peaks(ir_bgr, p) for p in peaks]
