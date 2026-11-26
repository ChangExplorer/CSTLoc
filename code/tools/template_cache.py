"""Deterministic DINOv2 template-feature cache used by locator inference.

Evaluation and candidate extraction consume per-template backbone features
(layer 7/15/23 maps + layer 15/23 GAP) so the template tower runs once per
template instead of once per trial. The cache is built from the dataset on
first use (no checkpoints needed) and stored at
code/diag_out/n_cache_tmpl_{split}.pt (gitignored).
"""
import time

import cv2
import torch

from models.encoders.dinov2_encoder import DINOv2TemplateEncoder
from tools.dataset_io import ROOT, list_instances
from tools.image_ops import norm224
from tools.paths import OUT

DEVICE = 'cuda'


def cache_path(split):
    return OUT / f'n_cache_tmpl_{split}.pt'


@torch.no_grad()
def build(split, force=False):
    """Encode every protocol template of a split and save the cache. Returns it."""
    out = cache_path(split)
    if out.exists() and not force:
        print(f'[cache] exists: {out} ({out.stat().st_size/1e9:.2f} GB)')
        return torch.load(out, map_location='cpu')
    enc = DINOv2TemplateEncoder(pretrained=True).to(DEVICE).eval()
    cache, t0 = {}, time.time()
    for inst, tmpls, _frames in list_instances(split):
        d = {}
        for s in range(0, len(tmpls), 32):
            chunk = tmpls[s:s + 32]
            xs = torch.stack([
                norm224(cv2.resize(cv2.cvtColor(
                    cv2.imread(str(ROOT / split / inst / 'rgb' / 'template' / f'{t:06d}.jpg')),
                    cv2.COLOR_BGR2RGB), (224, 224), interpolation=cv2.INTER_LINEAR))
                for t in chunk]).to(DEVICE)
            maps = enc.backbone.get_intermediate_layers(xs, n=[7, 15, 23], reshape=True)
            gaps = enc.backbone.get_intermediate_layers(xs, n=[15, 23], reshape=False)
            gap = torch.cat([o.mean(dim=1) for o in gaps], dim=1)
            for j, t in enumerate(chunk):
                d[str(t)] = {'m7': maps[0][j].cpu(), 'm15': maps[1][j].cpu(),
                             'm23': maps[2][j].cpu(), 'gap': gap[j].cpu()}
        cache[inst] = d
        print(f'  [cache] {inst}: {len(d)} templates ({time.time()-t0:.0f}s)', flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, out)
    print(f'[cache] wrote {out} ({out.stat().st_size/1e9:.2f} GB) in {time.time()-t0:.0f}s')
    return cache


def load(split):
    """Load the cache for a split, building it from the dataset if missing."""
    p = cache_path(split)
    if not p.exists():
        print(f'[cache] {p} missing; building from dataset (one-time, no checkpoints needed)')
        return build(split)
    return torch.load(p, map_location='cpu')
