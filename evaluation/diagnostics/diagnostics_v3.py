#!/usr/bin/env python
# evaluation/diagnostics_v3.py — v3 test 诊断：候选上限、hard rank-1、二分类失效、时间归一化曲线
import json, os, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np
import cv2

BASE = Path(os.environ.get('CSTLOC_ROOT', Path(__file__).resolve().parents[2]))
DATA = Path(os.environ.get('CSTLOC_DATA_ROOT', '/home/user/Code/Datasets/CSTLoc'))
OUT = BASE / 'evaluation/cache/diagnostics_v3'
OUT.mkdir(parents=True, exist_ok=True)
SPLIT = 'test'
IMG_H, IMG_W = 384, 640
TAU = 0.5
UNSEEN = {'animal', 'truck'}
KS_ORACLE = (1, 5, 10, 20)
TIME_EDGES = np.linspace(0, 1, 11)
K_BINS = [(0, 0), (1, 1), (2, 2), (3, 4), (5, 7), (8, 99)]

sys.path.insert(0, str(BASE / 'evaluation'))
from aggregate import pck_hit, aggregate  # noqa: E402


def read_frames():
    """返回 frame dict: (inst,f)->{gt,obj,tbox,distract,dboxes,diff}。"""
    frames = {}
    root = DATA / SPLIT
    for inst_dir in sorted(root.iterdir()):
        if not inst_dir.is_dir():
            continue
        mp = inst_dir / 'meta.json'
        if mp.exists() and json.load(open(mp, encoding='utf-8')).get('rgb_usable') is False:
            continue
        inst = inst_dir.name
        for jf in sorted((inst_dir / 'ir' / 'search').glob('*.json')):
            d = json.load(open(jf, encoding='utf-8'))
            tbox = None
            distract, dboxes = [], []
            for s in d.get('shapes', []):
                pts = np.asarray(s.get('points', []), dtype=float)
                if pts.size == 0:
                    continue
                xs, ys = pts[:, 0], pts[:, 1]
                box = [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]
                c = [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]
                wh = [box[2] - box[0], box[3] - box[1]]
                if (s.get('flags') or {}).get('target'):
                    tbox = (c, wh, box)
                else:
                    distract.append(c); dboxes.append(box)
            if tbox is None:
                continue
            gt, obj, box = tbox
            frames[(inst, int(jf.stem))] = {
                'gt': gt, 'obj': obj, 'tbox': box,
                'distract': distract, 'dboxes': dboxes,
                'diff': 'hard' if distract else 'easy',
            }
    return frames


def load_trials():
    src = BASE / 'evaluation/cache/v3_test_diag/main_results/results_full_s2024.jsonl'
    trials, key2idx = [], {}
    for ln in open(src, encoding='utf-8'):
        r = json.loads(ln)
        k = (r['inst'], int(r['t']), int(r['f']))
        assert k not in key2idx, k
        key2idx[k] = len(trials)
        trials.append(k)
    return trials, key2idx


def build_arrays(trials, frames):
    n = len(trials)
    gt = np.zeros((n, 2)); obj = np.zeros((n, 2))
    tbox = np.zeros((n, 4)); hard = np.zeros(n, bool)
    frame_idx = np.zeros(n, np.int32)
    insts = []
    fkeys = sorted(frames)
    fmap = {k: i for i, k in enumerate(fkeys)}
    dframes = [frames[k] for k in fkeys]
    for i, (inst, t, f) in enumerate(trials):
        fr = frames[(inst, f)]
        gt[i] = fr['gt']; obj[i] = fr['obj']; tbox[i] = fr['tbox']
        hard[i] = fr['diff'] == 'hard'
        frame_idx[i] = fmap[(inst, f)]
        insts.append(inst)
    # pair id
    pk = {}
    pair_id = np.zeros(n, np.int32)
    for i, (inst, t, f) in enumerate(trials):
        pair_id[i] = pk.setdefault((inst, t), len(pk))
    span = {}
    for (inst, t, f) in trials:
        span[inst] = max(span.get(inst, 0), abs(f - t))
    rho = np.array([abs(f - t) / span[inst] if span[inst] else 0.0
                    for (inst, t, f) in trials])
    kk = np.array([len(dframes[frame_idx[i]]['distract']) for i in range(n)])
    return dict(n=n, gt=gt, obj=obj, tbox=tbox, hard=hard,
                frame_idx=frame_idx, dframes=dframes, insts=insts,
                pair_id=pair_id, rho=rho, kk=kk, key2idx=None)


def pm(hit, pair_id, mask=None):
    h = hit if mask is None else hit[mask]
    p = pair_id if mask is None else pair_id[mask]
    if len(h) == 0:
        return None
    npair = int(p.max()) + 1
    s = np.bincount(p, weights=h.astype(float), minlength=npair)
    c = np.bincount(p, minlength=npair)
    ok = c > 0
    return float(np.mean(s[ok] / c[ok]))


def load_pred(path, key2idx, n):
    pred = np.full((n, 2), np.nan)
    for ln in open(path, encoding='utf-8'):
        r = json.loads(ln)
        i = key2idx[(r['inst'], int(r['t']), int(r['f']))]
        pred[i] = [float(r['pred'][0]), float(r['pred'][1])]
    assert np.isfinite(pred).all(), path
    return pred


def hit_of(pred, A):
    d = np.hypot(pred[:, 0] - A['gt'][:, 0], pred[:, 1] - A['gt'][:, 1])
    return d < TAU * np.max(A['obj'], axis=1)


def in_box(p, box):
    return box[0] <= p[0] <= box[2] and box[1] <= p[1] <= box[3]


def box_dist(p, box):
    dx = max(box[0] - p[0], 0.0, p[0] - box[2])
    dy = max(box[1] - p[1], 0.0, p[1] - box[3])
    return float(np.hypot(dx, dy))


def block(pred, A):
    n = A['n']; hit = hit_of(pred, A)
    out = {'overall': pm(hit, A['pair_id']),
           'easy': pm(hit, A['pair_id'], ~A['hard']),
           'hard': pm(hit, A['pair_id'], A['hard']),
           'seen': pm(hit, A['pair_id'], np.array([s.rsplit('_', 1)[0] not in UNSEEN for s in A['insts']])),
           'unseen': pm(hit, A['pair_id'], np.array([s.rsplit('_', 1)[0] in UNSEEN for s in A['insts']])),
           'n_trials': n, 'hit_rate_micro': float(hit.mean())}
    # hard rank-1 / MRR：预测点分配给“距离最近的同类 GT 矩形框”（框内距离 0）
    hidx = np.where(A['hard'])[0]
    ok = 0; okc = 0; rr = 0.0; inst_err = 0; bg_err = 0; nfail = 0
    in_any = 0; in_tgt = 0; in_dis = 0
    for i in hidx:
        fr = A['dframes'][A['frame_idx'][i]]
        p = pred[i]
        dt = box_dist(p, fr['tbox'])
        ds = [box_dist(p, b) for b in fr['dboxes']]
        rank = 1 + int(sum(d < dt for d in ds))
        ok += int(rank == 1); rr += 1.0 / rank
        if in_box(p, fr['tbox']):
            in_any += 1; in_tgt += 1
        elif any(in_box(p, b) for b in fr['dboxes']):
            in_any += 1; in_dis += 1
        # 兼容旧中心点口径（仅报告，不用于论文）
        ct = float(np.hypot(p[0] - A['gt'][i, 0], p[1] - A['gt'][i, 1]))
        cd = [float(np.hypot(p[0] - c[0], p[1] - c[1])) for c in fr['distract']]
        okc += int(1 + sum(d < ct for d in cd) == 1)
    out['rank1_hard'] = float(ok / len(hidx)) if len(hidx) else None
    out['mrr_hard'] = float(rr / len(hidx)) if len(hidx) else None
    out['rank1_hard_center'] = float(okc / len(hidx)) if len(hidx) else None
    out['rank1_inbox'] = float(in_tgt / in_any) if in_any else None
    out['rank1_inbox_n'] = int(in_any)
    out['rank1_inbox_coverage'] = float(in_any / len(hidx)) if len(hidx) else None
    out['in_target_box'] = float(in_tgt / len(hidx)) if len(hidx) else None
    out['in_distractor_box'] = float(in_dis / len(hidx)) if len(hidx) else None
    # 论文使用的实例 rank-1：预测点归属的同类实例框是目标框
    out['rank1_instance'] = out['in_target_box']
    # 二分类失效（全部未命中 trial，含 easy 背景帧；hard 帧才有干扰物框）
    for i in range(n):
        if hit[i]:
            continue
        nfail += 1
        fr = A['dframes'][A['frame_idx'][i]]
        p = pred[i]
        if any(in_box(p, b) for b in fr['dboxes']):
            inst_err += 1
        else:
            bg_err += 1
    out['fail_n'] = nfail
    out['fail_instance'] = float(inst_err / nfail) if nfail else None
    out['fail_background'] = float(bg_err / nfail) if nfail else None
    # hard-only 失效构成
    hf_i = hf_b = hf_n = 0
    for i in hidx:
        if hit[i]:
            continue
        hf_n += 1
        fr = A['dframes'][A['frame_idx'][i]]
        if any(in_box(pred[i], b) for b in fr['dboxes']):
            hf_i += 1
        else:
            hf_b += 1
    out['fail_n_hard'] = hf_n
    out['fail_instance_hard'] = float(hf_i / hf_n) if hf_n else None
    out['fail_background_hard'] = float(hf_b / hf_n) if hf_n else None
    # 时间分位
    tb = np.clip((A['rho'] * 10).astype(int), 0, 9)
    out['time_curve'] = []
    for b in range(10):
        m = tb == b
        out['time_curve'].append({'bin': b, 'rho_mid': float((TIME_EDGES[b] + TIME_EDGES[b + 1]) / 2),
                                  'n': int(m.sum()), 'pm': pm(hit, A['pair_id'], m),
                                  'micro': float(hit[m].mean()) if m.any() else None})
    # 干扰物数量分档
    out['k_curve'] = []
    for lo, hi in K_BINS:
        m = (A['kk'] >= lo) & (A['kk'] <= hi)
        out['k_curve'].append({'lo': lo, 'hi': hi, 'n': int(m.sum()),
                               'pm': pm(hit, A['pair_id'], m),
                               'micro': float(hit[m].mean()) if m.any() else None})
    return out, hit


def lb_params(inst, f, cache):
    k = (inst, f)
    if k in cache:
        return cache[k]
    p = DATA / SPLIT / inst / 'ir/search' / f'{f:06d}.jpg'
    img = cv2.imread(str(p))
    h, w = img.shape[:2]
    scale = min(IMG_W / w, IMG_H / h)
    nw, nh = round(w * scale), round(h * scale)
    cache[k] = (scale, (IMG_W - nw) // 2, (IMG_H - nh) // 2)
    return cache[k]


def oracle_from_scores(seed, A, key2idx):
    n = A['n']
    flags = {K: np.zeros(n, bool) for K in KS_ORACLE}
    src = BASE / f'code/diag_out/tcia_v2_s{seed}_test_tcia_scores'
    cache = {}
    t0 = time.time()
    for f in sorted(src.glob('*.jsonl')):
        inst = f.stem
        for ln in open(f, encoding='utf-8'):
            r = json.loads(ln)
            i = key2idx[(inst, int(r['t']), int(r['f']))]
            scale, pl, pt = lb_params(inst, int(r['f']), cache)
            pk = np.asarray(r['peaks'][:max(KS_ORACLE)], dtype=float)
            pts = (pk[:, :2] - np.array([pl, pt])) / scale
            d = np.hypot(pts[:, 0] - A['gt'][i, 0], pts[:, 1] - A['gt'][i, 1])
            th = TAU * max(A['obj'][i])
            for K in KS_ORACLE:
                flags[K][i] = bool(np.any(d[:K] < th))
    out = {}
    for K in KS_ORACLE:
        h = flags[K]
        out[f'oracle@{K}'] = {
            'pair_macro': pm(h, A['pair_id']), 'micro': float(h.mean()),
            'easy': pm(h, A['pair_id'], ~A['hard']), 'hard': pm(h, A['pair_id'], A['hard'])}
    print(f'oracle seed {seed} done {time.time()-t0:.1f}s', flush=True)
    return out, flags


def main():
    t0 = time.time()
    frames = read_frames()
    print('frames', len(frames), flush=True)
    trials, key2idx = load_trials()
    A = build_arrays(trials, frames)
    A['key2idx'] = key2idx
    print('trials', A['n'], 'pairs', int(A['pair_id'].max()) + 1,
          'hard trials', int(A['hard'].sum()), flush=True)

    SYS = [
        ('Full', ['evaluation/cache/v3_test_diag/main_results/results_full_s42.jsonl',
                  'evaluation/cache/v3_test_diag/main_results/results_full_s123.jsonl',
                  'evaluation/cache/v3_test_diag/main_results/results_full_s2024.jsonl']),
        ('Locator', ['evaluation/cache/v3_test_diag/main_results/results_locator_s2024.jsonl']),
        ('PerSAM', ['evaluation/cache/v3_comparison/persam.results.jsonl']),
        ('NCC', ['evaluation/cache/v3_comparison/ncc.results.jsonl']),
        ('RoMa', ['evaluation/cache/v3_comparison/roma.results.jsonl']),
        ('DINOv2', ['evaluation/cache/v3_comparison/dinov2.results.jsonl']),
        ('Grounding DINO', ['evaluation/cache/v3_comparison/grounding.results.jsonl']),
    ]
    result = {'meta': {'split': SPLIT, 'n_trials': A['n'], 'n_pairs': int(A['pair_id'].max()) + 1,
                       'n_hard': int(A['hard'].sum()), 'data_root': str(DATA)},
              'systems': {}, 'full_seeds': {}}
    for name, files in SYS:
        per = []
        for fp in files:
            pred = load_pred(BASE / fp, key2idx, A['n'])
            b, hit = block(pred, A)
            per.append(b)
        if len(per) == 1:
            result['systems'][name] = per[0]
        else:
            agg = {}
            keys = [k for k in per[0] if k not in ('time_curve', 'k_curve')]
            for k in keys:
                vals = [p[k] for p in per if p.get(k) is not None]
                agg[k] = float(np.mean(vals)) if vals and isinstance(vals[0], (int, float)) else vals[0]
            agg['time_curve'] = []
            for j in range(10):
                vals = [p['time_curve'][j]['pm'] for p in per if p['time_curve'][j]['pm'] is not None]
                agg['time_curve'].append({'bin': j, 'rho_mid': per[0]['time_curve'][j]['rho_mid'],
                                          'n': per[0]['time_curve'][j]['n'],
                                          'pm': float(np.mean(vals)) if vals else None})
            agg['k_curve'] = []
            for j in range(len(K_BINS)):
                vals = [p['k_curve'][j]['pm'] for p in per if p['k_curve'][j]['pm'] is not None]
                agg['k_curve'].append({'lo': per[0]['k_curve'][j]['lo'], 'hi': per[0]['k_curve'][j]['hi'],
                                       'n': per[0]['k_curve'][j]['n'],
                                       'pm': float(np.mean(vals)) if vals else None})
            result['systems'][name] = agg
            result['full_seeds'][name] = per
        print('system', name, result['systems'][name]['overall'], flush=True)

    # oracle 候选上限（三 seed）
    ores = []
    for seed in (42, 123, 2024):
        o, flags = oracle_from_scores(seed, A, key2idx)
        ores.append(o)
    orc = {}
    for K in KS_ORACLE:
        key = f'oracle@{K}'
        orc[key] = {kk: float(np.mean([o[key][kk] for o in ores]))
                    for kk in ('pair_macro', 'micro', 'easy', 'hard')}
    result['oracle'] = orc
    result['oracle_seeds'] = ores
    print('oracle', json.dumps(orc, ensure_ascii=False), flush=True)
    (OUT / 'diagnostics_v3.json').write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding='utf-8')
    print('wrote', OUT / 'diagnostics_v3.json', 'elapsed', time.time() - t0, flush=True)


if __name__ == '__main__':
    main()
