# TCIA 完整 val/test 选择器：top-20 峰 + 224 crop 管线，余弦 argmax。
# hit=hypot(peak-gt)<=0.5*max(obj)；pair-macro=(inst,t)帧均再模板宏均。
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))

from models.tcia import TCIAv2
from tools import template_cache
from tools.dataset_io import (ROOT, letterboxed_bgr, load_frame,
                              load_template_tensor, protocol_frame_trials,
                              template_ids)
from tools.image_ops import norm224
from tools.locator import build_locator, extract_peaks_and_crops
from tools.paths import OUT, weights_dir, write_jsonl

DEVICE = 'cuda'
SEED = int(os.environ.get('TCIA_SEED', '0'))
VAL_TOTAL = int(os.environ.get('TCIA_VAL_TOTAL', '171587'))
TEST_TOTAL = int(os.environ.get('TCIA_TEST_TOTAL', '327772'))
UNSEEN_PREFIX = ('animal_', 'truck_')


def default_tcia_ckpt() -> Path:
    v = os.environ.get('TCIA_BEST_CKPT')
    if v:
        return Path(v)
    return weights_dir() / 'tcia' / 'best_fullval.pt'


def score_dir_for(split):
    d = OUT / f'tcia_{split}_scores'
    d.mkdir(parents=True, exist_ok=True)
    return d


def hit_peak(peak, gt, obj):
    return bool(np.hypot(float(peak[0]) - gt[0], float(peak[1]) - gt[1]) <= 0.5 * max(obj))


def pair_macro(rows):
    out = {}
    for key in ('hit_tcia', 'hit_locator'):
        def stat(sub):
            if not sub:
                return {'pair_macro': None, 'n_pairs': 0, 'n_trials': 0}
            groups = {}
            for r in sub:
                groups.setdefault((r['inst'], r['t']), []).append(int(r[key]))
            return {'pair_macro': float(np.mean([np.mean(v) for v in groups.values()])),
                    'n_pairs': len(groups), 'n_trials': len(sub)}
        unseen = [r for r in rows if r['inst'].startswith(UNSEEN_PREFIX)]
        seen = [r for r in rows if not r['inst'].startswith(UNSEEN_PREFIX)]
        out[key] = {'overall': stat(rows), 'easy': stat([r for r in rows if r['diff'] == 'easy']),
                    'hard': stat([r for r in rows if r['diff'] == 'hard']),
                    'seen': stat(seen), 'unseen': stat(unseen)}
    return out


def _assert_env():
    assert torch.cuda.is_available(), 'CUDA GPU required'
    free = int(os.popen("free -m | awk 'NR==2{print $7}'").read().strip())
    assert free >= 4000, free


@torch.no_grad()
def run_split(model, split, instances=None, write_scores=True):
    """TCIA-单独 pass。model: TCIAv2（eval 模式，已在 DEVICE）。
    返回 (summary, rows)。rows 逐 trial 含 hit_tcia / hit_locator。"""
    assert split in ('val', 'test')
    _assert_env()
    locator = build_locator()
    expected_total = VAL_TOTAL if split == 'val' else TEST_TOTAL
    tmpl_cache = template_cache.load(split)
    all_insts = sorted(p.name for p in (ROOT / split).iterdir()
                       if p.is_dir() and template_ids(split, p.name))
    all_insts = [i for i in all_insts
                 if not ((ROOT / split / i / 'meta.json').exists() and
                         json.loads((ROOT / split / i / 'meta.json').read_text(encoding='utf-8')).get('rgb_usable') is False)]
    if instances is not None:
        keep = set(instances)
        all_insts = [i for i in all_insts if i in keep]
        expected_total = None
    rows, t0, total_done = [], time.time(), 0
    sdir = score_dir_for(split) if write_scores else None
    for inst in all_insts:
        tmpls = template_ids(split, inst)
        frames = sorted(int(p.stem) for p in (ROOT / split / inst / 'ir/search').glob('*.json'))
        expected_inst = sum(len(protocol_frame_trials(split, inst, f, tmpls)) for f in frames)
        cache = tmpl_cache[inst]
        tz = {}
        for s in range(0, len(tmpls), 32):
            ts = tmpls[s:s + 32]
            x = torch.stack([load_template_tensor(split, inst, t) for t in ts]).to(DEVICE)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                z = model.encode_head(x, 'tmpl').float()
            for j, t in enumerate(ts):
                tz[t] = z[j]
        inst_rows = []
        for f in frames:
            trials = protocol_frame_trials(split, inst, f, tmpls)
            fr = load_frame(ROOT / split / inst, f)
            ir_bgr = letterboxed_bgr(split, inst, f)
            gt = [float(v) for v in fr['gt_center']]
            obj = [float(v) for v in fr['obj_size']]
            trial_data = []
            for s in range(0, len(trials), 12):
                sub = trials[s:s + 12]
                peak_list, crop_list = extract_peaks_and_crops(
                    locator, [cache[str(r['t'])] for r in sub], fr, ir_bgr)
                for r, peaks, crops in zip(sub, peak_list, crop_list):
                    trial_data.append((r, peaks, crops))
            uniq, seen = [], {}
            for r, peaks, crops in trial_data:
                for (x, y, _v), crop in zip(peaks, crops):
                    key = (int(x), int(y))
                    if key not in seen:
                        seen[key] = len(uniq)
                        uniq.append((key, crop))
            zs = []
            for s in range(0, len(uniq), 128):
                x = torch.stack([norm224(c) for _k, c in uniq[s:s + 128]]).to(DEVICE)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    zs.append(model.encode_head(x, 'cand').float())
            zs = torch.cat(zs) if zs else torch.empty(0, 256, device=DEVICE)
            for r, peaks, crops in trial_data:
                idx = torch.tensor([seen[(int(x), int(y))] for x, y, _v in peaks], device=DEVICE)
                sc = (zs[idx] @ tz[r['t']]).cpu().numpy()
                pi = int(np.argmax(sc))
                inst_rows.append({'inst': inst, 't': r['t'], 'f': f, 'diff': fr['difficulty'],
                                  'pred_lb': [float(peaks[pi][0]), float(peaks[pi][1])],
                                  'peaks': [[int(x), int(y), round(float(v), 7)] for x, y, v in peaks],
                                  'hit_tcia': hit_peak(peaks[pi], gt, obj),
                                  'hit_locator': hit_peak(peaks[0], gt, obj)})
            del trial_data, uniq, zs
        assert len(inst_rows) == expected_inst, (inst, len(inst_rows), expected_inst)
        rows.extend(inst_rows)
        total_done += len(inst_rows)
        print(f'[{split} {inst}] DONE trials={len(inst_rows)} total={total_done} '
              f'elapsed={time.time() - t0:.0f}s', flush=True)
        if sdir is not None:
            write_jsonl(sdir / f'{inst}.jsonl', inst_rows)
        del cache, tz
    if expected_total is not None:
        assert total_done == expected_total, (total_done, expected_total)
    metrics = pair_macro(rows)
    flat = {}
    for k in ('hit_tcia', 'hit_locator'):
        for m in ('overall', 'easy', 'hard', 'seen', 'unseen'):
            flat[f'{k}_{m}'] = metrics[k][m]['pair_macro']
    summary = {'type': f'{split}_tcia_only', 'seed': SEED, 'n_trials': total_done,
               'elapsed_s': time.time() - t0, **flat, 'detail': metrics}
    return summary, rows


def load_adapter(ck):
    state = torch.load(ck, map_location='cpu')
    model = TCIAv2().to(DEVICE).eval()
    model.load_adapter_state_dict(state['state'])
    return model, state


def final_test(ck=None):
    if ck is None:
        ck = default_tcia_ckpt()
    ck = Path(ck)
    assert ck.exists(), f'TCIA checkpoint not found: {ck}'
    model, _state = load_adapter(ck)
    summary, _rows = run_split(model, 'test')
    write_jsonl(OUT / f'tcia_s{SEED}_test_summary.jsonl', [summary])
    print('TEST_TCIA_ONLY', json.dumps({k: summary[k] for k in
          ('hit_tcia_overall', 'hit_tcia_hard', 'hit_tcia_seen', 'hit_tcia_unseen',
           'hit_locator_overall', 'n_trials', 'elapsed_s')}, ensure_ascii=False), flush=True)
    return summary


def self_test():
    assert hit_peak([3, 4], [0, 0], [8, 6]) is False
    assert hit_peak([3, 4], [0, 0], [12, 6]) is True
    rows = [{'inst': 'a', 't': 1, 'diff': 'hard', 'hit_tcia': True, 'hit_locator': False},
            {'inst': 'a', 't': 1, 'diff': 'hard', 'hit_tcia': False, 'hit_locator': False},
            {'inst': 'a', 't': 2, 'diff': 'easy', 'hit_tcia': True, 'hit_locator': True},
            {'inst': 'b', 't': 1, 'diff': 'easy', 'hit_tcia': False, 'hit_locator': True}]
    m = pair_macro(rows)
    assert abs(m['hit_tcia']['overall']['pair_macro'] - 0.5) < 1e-12
    assert abs(m['hit_tcia']['hard']['pair_macro'] - 0.5) < 1e-12
    assert abs(m['hit_locator']['overall']['pair_macro'] - (2 / 3)) < 1e-12
    print('TCIA_EVAL_SELF_TEST_PASS', SEED)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', required=True, choices=['self_test', 'test'])
    ap.add_argument('--ckpt', default=None, help='TCIA checkpoint (default: TCIA_BEST_CKPT '
                                                 'or weights/tcia/best_fullval.pt)')
    args = ap.parse_args()
    if args.phase == 'self_test':
        self_test()
    else:
        final_test(ck=args.ckpt)


if __name__ == '__main__':
    main()
