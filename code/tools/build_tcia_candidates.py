#!/usr/bin/env python3
"""Build TCIA-v2 training candidates from a locator checkpoint on the v3 dataset.

The public release ships the candidate file. For a full from-scratch run the
locator changes, so candidates must be regenerated. This script samples the v3
train pairs, runs the locator, keeps trials with at least one peak inside the GT
box, and writes the same schema consumed by tools/tcia_v2_train.py.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ['CSTLOC_PROJ_ADAPTER'] = 'orth'

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from tools.cmi_probe_build import build_locator, label_peaks  # noqa: E402
from tools.tcia_data import (ROOT, frame_ids, load_template_tensor, sha256,
                             template_ids, train_donor_for, write_jsonl)  # noqa: E402
from tools.v_verifier_rerank import MIN_DIST, nms_peaks  # noqa: E402
from tools.wrong_template_probe import load_frame  # noqa: E402

DEVICE = 'cuda'
K = 20
BATCH_T = 12
MIN_GAP = 30


def pipeline_sha():
    h = hashlib.sha256()
    for name in ('tools/build_tcia_candidates.py', 'tools/cmi_probe_build.py',
                 'tools/tcia_v2_data.py', 'tools/wrong_template_probe.py',
                 'tools/v_verifier_rerank.py', 'models/localizer.py'):
        p = CODE / name
        h.update(name.encode())
        h.update(bytes.fromhex(sha256(p)))
    h.update(f'|K={K}|min_dist={MIN_DIST}|BATCH_T={BATCH_T}|MIN_GAP={MIN_GAP}'.encode())
    return h.hexdigest()


def groups_for_seed(seed):
    """Enumerate eligible (inst, f, t), shuffle trials, group by frame."""
    recs = []
    insts = sorted(p.name for p in (ROOT / 'train').iterdir() if p.is_dir())
    for inst in insts:
        ts = template_ids('train', inst)
        fs = frame_ids('train', inst)
        if not ts or not fs:
            continue
        for f in fs:
            for t in ts:
                if abs(f - t) >= MIN_GAP:
                    recs.append((inst, f, t))
    if not recs:
        raise SystemExit(f'no eligible trials under {ROOT}')
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(recs))
    groups, order = {}, []
    for idx in perm:
        inst, f, t = recs[int(idx)]
        key = (inst, f)
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(t)
    return insts, recs, groups, order, rng


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--locator', required=True)
    ap.add_argument('--n-trials', type=int, default=20000)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix('.partial.jsonl')
    prog = out.with_suffix('.progress.json')

    loc_sha = sha256(args.locator)
    insts, recs, groups, order, rng = groups_for_seed(args.seed)
    print(f'[cand] train instances={len(insts)} eligible_trials={len(recs)} '
          f'frames={len(order)} locator_sha={loc_sha[:16]}', flush=True)

    model = build_locator()
    donor_map = {}
    for inst in insts:
        try:
            donor_map[inst] = train_donor_for(inst)
        except AssertionError:
            print(f'[cand] skip {inst}: no same-class donor with templates', flush=True)
    if not donor_map:
        raise SystemExit('no train instance with a valid donor')
    rows, scanned, t0 = [], 0, time.time()
    last_flush = 0

    for inst, f in order:
        if len(rows) >= args.n_trials:
            break
        if inst not in donor_map:
            continue
        ts = groups[(inst, f)]
        if len(ts) > BATCH_T:
            pick = rng.choice(len(ts), BATCH_T, replace=False)
            ts = [ts[int(i)] for i in sorted(pick)]
        fr = load_frame(ROOT / 'train' / inst, f)
        ir_x = fr['ir'].unsqueeze(0).to(DEVICE)
        tmpls = torch.stack([load_template_tensor('train', inst, t) for t in ts]).unsqueeze(1).to(DEVICE)
        with torch.no_grad():
            ir_feats = model.ir_encoder(ir_x)
            b = len(ts)
            ir_feats = tuple(z.expand(b, -1, -1, -1) for z in ir_feats)
            logits = model(ir_x.expand(b, -1, -1, -1), tmpls, ir_feats=ir_feats)
        heats = torch.sigmoid(logits)[:, 0].cpu().numpy()
        donor, tmap = donor_map[inst]
        box = [float(v) for v in fr['gt_box']]
        gt = [(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0]
        obj = [box[2] - box[0], box[3] - box[1]]
        for t, heat in zip(ts, heats):
            peaks = nms_peaks(heat, K, MIN_DIST)
            labels = label_peaks(peaks, fr['gt_box'], fr['distractor_boxes'])
            if 2 not in labels:
                continue
            pos = labels.index(2)
            if pos != 0:
                order = [pos] + [i for i in range(len(labels)) if i != pos]
                peaks = [peaks[i] for i in order]
                labels = [labels[i] for i in order]
            rows.append({
                'schema': 'tcia_v2.peakcache.v1', 'split': 'train',
                'inst': inst, 't': int(t), 'f': int(f),
                'diff': fr['difficulty'], 'gt': gt, 'obj': obj,
                'peaks': [[int(round(float(x))), int(round(float(y))), float(v)]
                          for x, y, v in peaks],
                'labels': labels,
                'donor': {'split': 'train', 'inst': donor, 't': int(tmap[t])},
                'source': {'origin': 'v3_from_locator', 'line': None,
                           'pool_index': None, 'locator_sha256': loc_sha},
            })
            if len(rows) >= args.n_trials:
                break
        scanned += 1
        if scanned % 100 == 0 or len(rows) >= args.n_trials:
            dt = time.time() - t0
            print(f'[cand] frames={scanned}/{len(order)} rows={len(rows)} '
                  f'rate={len(rows)/max(dt,1e-9):.2f}/s elapsed={dt:.0f}s', flush=True)
        if len(rows) >= last_flush + 1000:
            write_jsonl(tmp, rows)
            prog.write_text(json.dumps({'rows': len(rows), 'frames': scanned},
                                       ensure_ascii=False), encoding='utf-8')
            last_flush = len(rows)

    if len(rows) < args.n_trials:
        raise SystemExit(f'only {len(rows)} positive trials from {scanned} frames'
                         f' (wanted {args.n_trials})')

    write_jsonl(out, rows)
    manifest = {
        'schema': 'tcia_v2.candidates.manifest.v1',
        'origin': 'v3_from_locator',
        'dataset_root': str(ROOT),
        'locator': str(args.locator),
        'locator_sha256': loc_sha,
        'seed': args.seed,
        'n_trials': len(rows),
        'frames_scanned': scanned,
        'pipeline_sha256': pipeline_sha(),
        'candidates_sha256': sha256(out),
    }
    out.with_suffix('.manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    if tmp.exists():
        tmp.unlink()
    if prog.exists():
        prog.unlink()
    print(f'[cand] DONE {out} rows={len(rows)} sha256={manifest["candidates_sha256"]}',
          flush=True)


if __name__ == '__main__':
    main()
