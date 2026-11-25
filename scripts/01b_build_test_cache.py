#!/usr/bin/env python
"""Build the deterministic DINOv2 template cache required by the v3 test evaluator.

Creates code/diag_out/n_cache_tmpl_test.pt from the dataset (no checkpoints needed).
Run once after 01_prepare_dataset.sh; takes a few minutes on a single GPU.
"""
import os, sys, time
from pathlib import Path
os.environ.setdefault('CSTLOC_SPLIT', 'test')
CODE = Path(__file__).resolve().parents[1] / 'code'
sys.path.insert(0, str(CODE))
import cv2, torch
import tools.n_single_template_eval as n
from models.encoders.dinov2_encoder import DINOv2TemplateEncoder
from tools.v_verifier_rerank import norm224, DEVICE

OUT = n.OUT_DIR / 'n_cache_tmpl_test.pt'
if OUT.exists():
    print(f'[cache] exists: {OUT} ({OUT.stat().st_size/1e9:.2f} GB)')
    sys.exit(0)
enc = DINOv2TemplateEncoder(pretrained=True).to(DEVICE).eval()
cache, t0 = {}, time.time()
with torch.no_grad():
    for inst, tmpls, _ in n.val_instances():
        d = {}
        for s in range(0, len(tmpls), 32):
            chunk = tmpls[s:s+32]
            xs = torch.stack([
                norm224(cv2.resize(cv2.cvtColor(
                    cv2.imread(str(n.split_dir()/inst/'rgb'/'template'/f'{t:06d}.jpg')),
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
OUT.parent.mkdir(parents=True, exist_ok=True)
torch.save(cache, OUT)
print(f'[cache] wrote {OUT} ({OUT.stat().st_size/1e9:.2f} GB) in {time.time()-t0:.0f}s')
