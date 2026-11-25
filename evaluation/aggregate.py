# evaluation/aggregate.py —— 统一评测聚合器
#
# 输入：results.jsonl（每行 {"inst","tmpl_mod","t","scene_mod","f","pred":[x,y]}，原始 1920×1080 坐标）
#       可选 peaks.jsonl（{"inst","t","f","peaks":[[x,y,score],...]}）→ oracle@K
# 输出：指标全表（打印 + --out 落盘 JSON）。
# 纪律：产物只存原始预测；GT/难度/类别在聚合时从数据集挂接（evaluation/cache/ 帧表缓存）。
#       对外口径一律 test；val 仅用于训练期选型与本链验收。
import argparse
import json
import os
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

EVAL_DIR = Path(__file__).resolve().parent
CACHE = EVAL_DIR / 'cache'
UNSEEN = {'animal', 'truck'}
TAU_SCAN = [round(0.1 * i, 1) for i in range(1, 11)]


def _read_frame(args):
    """读一个场景帧 json → (gt_center, obj_hw, n_distractor, distractor_centers)。"""
    path, = args
    d = json.load(open(path, encoding='utf-8'))
    gt = obj = None
    distract = []
    for s in d.get('shapes', []):
        pts = np.asarray(s.get('points', []), dtype=float)
        if pts.size == 0:
            continue
        xs, ys = pts[:, 0], pts[:, 1]
        c = [float(xs.mean()), float(ys.mean())]
        wh = [float(xs.max() - xs.min()), float(ys.max() - ys.min())]
        if (s.get('flags') or {}).get('target'):
            gt, obj = c, wh
        else:
            distract.append(c)
    return gt, obj, distract


def build_frame_table(data_root, split):
    """(inst,f) → {gt, obj, diff, distract}；带缓存。"""
    CACHE.mkdir(exist_ok=True)
    import hashlib
    root_key = hashlib.sha256(str(Path(data_root).resolve()).encode()).hexdigest()[:8]
    cache_f = CACHE / f'frame_table_{split}_{root_key}.json'
    if cache_f.exists():
        return {tuple(json.loads(k)): v for k, v in
                json.load(open(cache_f, encoding='utf-8')).items()}
    root = Path(data_root) / split
    jobs = []
    for inst_dir in sorted(root.iterdir()):
        if not inst_dir.is_dir():
            continue
        mp = inst_dir / 'meta.json'
        if mp.exists() and json.load(open(mp, encoding='utf-8')).get('rgb_usable') is False:
            continue
        for jf in sorted((inst_dir / 'ir' / 'search').glob('*.json')):
            jobs.append((str(jf),))
    with Pool(8) as pool:
        res = pool.map(_read_frame, jobs)
    table = {}
    for (jf,), (gt, obj, distract) in zip(jobs, res):
        p = Path(jf)
        inst = p.parents[2].name
        f_no = int(p.stem)
        if gt is None:
            continue
        table[(inst, f_no)] = {'gt': gt, 'obj': obj,
                               'diff': 'hard' if distract else 'easy',
                               'distract': distract}
    json.dump({json.dumps(k): v for k, v in table.items()},
              open(cache_f, 'w', encoding='utf-8'))
    return table


def pck_hit(pred, gt, obj, tau=0.5, mode='max'):
    th = tau * (max(obj) if mode == 'max' else min(obj))
    return float(((pred[0] - gt[0]) ** 2 + (pred[1] - gt[1]) ** 2) ** 0.5 < th)


def _pair_macro(hits):
    pair = defaultdict(list)
    for h, key in hits:
        pair[key].append(h)
    return float(np.mean([np.mean(v) for v in pair.values()]))


def aggregate(recs, peaks=None):
    """recs: list of dict(inst,t,f,gt,obj,diff,pred,dist,cls)。全部指标。"""
    def pm(sub):
        return round(_pair_macro([(r['hit'], (r['inst'], r['t'])) for r in sub]), 4) if sub else None

    def tm(sub):
        return round(float(np.mean([r['hit'] for r in sub])), 4) if sub else None

    out = {'n_trials': len(recs),
           'overall': {'pair_macro': pm(recs), 'trial_micro': tm(recs)},
           'easy': pm([r for r in recs if r['diff'] == 'easy']),
           'hard': pm([r for r in recs if r['diff'] == 'hard']),
           'seen': pm([r for r in recs if r['cls'] not in UNSEEN]),
           'unseen': pm([r for r in recs if r['cls'] in UNSEEN]),
           'pck_min': pm([dict(r, hit=r['hit_min']) for r in recs]),
           'auc_max': None}
    # AUC（τ·max 扫描，按对宏）
    pair = defaultdict(list)
    for r in recs:
        pair[(r['inst'], r['t'])].append(r['dist'] / max(r['obj']))
    out['auc_max'] = round(float(np.mean([
        np.mean([np.mean([v < t for v in vs]) for vs in pair.values()]) for t in TAU_SCAN])), 4)
    # per-class（对宏）
    pc = {}
    for c in sorted({r['cls'] for r in recs}):
        sub = [r for r in recs if r['cls'] == c]
        pc[c] = {'pair_macro': pm(sub), 'n': len(sub), 'unseen': c in UNSEEN}
    out['per_class'] = pc
    # 实例判别 rank-1（hard 子集；干扰物标注完工前为临时口径）：pred 最近邻 GT 中心是目标本身
    disc = [r for r in recs if r['diff'] == 'hard' and r['distract']]
    if disc:
        ok = 0
        for r in disc:
            cands = [r['gt']] + r['distract']
            d = [((r['pred'][0] - c[0]) ** 2 + (r['pred'][1] - c[1]) ** 2) for c in cands]
            ok += int(np.argmin(d) == 0)
        out['instance_rank1_hard_provisional'] = round(ok / len(disc), 4)
    # 失效三分类（临时口径）：漏检 = 未命中且预测距目标 >2×阈；实例混淆 = 未命中但距某干扰物 <阈；其余为模态/背景混淆
    cat = {'miss': 0, 'instance_confusion': 0, 'other_confusion': 0}
    for r in recs:
        if r['hit']:
            continue
        th = 0.5 * max(r['obj'])
        d_gt = ((r['pred'][0] - r['gt'][0]) ** 2 + (r['pred'][1] - r['gt'][1]) ** 2) ** 0.5
        d_dis = min([((r['pred'][0] - c[0]) ** 2 + (r['pred'][1] - c[1]) ** 2) ** 0.5
                     for c in r['distract']], default=1e18)
        if d_dis < th:
            cat['instance_confusion'] += 1
        elif d_gt > 2 * th:
            cat['miss'] += 1
        else:
            cat['other_confusion'] += 1
    n_miss = sum(cat.values())
    out['failure_taxonomy_provisional'] = {k: round(v / n_miss, 4) for k, v in cat.items()} if n_miss else None
    # 时间归一化：|f−t| / 实例内最大 |f−t|，三分位桶
    span = defaultdict(int)
    for r in recs:
        span[r['inst']] = max(span[r['inst']], abs(r['f'] - r['t']))
    nb = defaultdict(list)
    for r in recs:
        rho = abs(r['f'] - r['t']) / span[r['inst']] if span[r['inst']] else 0.0
        b = 0 if rho < 1 / 3 else (1 if rho < 2 / 3 else 2)
        nb[b].append((r['hit'], (r['inst'], r['t'])))
    out['time_norm_thirds'] = {('short', 'mid', 'long')[b]: round(_pair_macro(v), 4)
                               for b, v in sorted(nb.items())}
    # oracle@K（若有峰）
    if peaks:
        for K in (5, 20):
            o = [any(pck_hit([p[0], p[1]], r['gt'], r['obj']) for p in r['peaks'][:K]) for r in peaks]
            out[f'oracle@{K}'] = round(float(np.mean(o)), 4) if o else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', required=True)
    ap.add_argument('--peaks', default=None)
    ap.add_argument('--split', required=True, choices=['val', 'test'])
    ap.add_argument('--data-root', default=os.environ.get('CSTLOC_DATA_ROOT', str(EVAL_DIR.parents[1] / 'Datasets' / 'CSTLoc-UAV_v3')))
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    if args.split != 'test':
        print('[口径纪律] 当前聚合 split=%s：val 结果仅用于训练期参考与链路验收，对外一律 test。'
              % args.split, file=sys.stderr)

    table = build_frame_table(args.data_root, args.split)
    recs = []
    miss_gt = 0
    for line in open(args.results, encoding='utf-8'):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        key = (r['inst'], int(r['f']))
        fr = table.get(key)
        if fr is None:
            miss_gt += 1
            continue
        pred = [float(r['pred'][0]), float(r['pred'][1])]
        recs.append({'inst': r['inst'], 't': int(r['t']), 'f': int(r['f']),
                     'cls': r['inst'].rsplit('_', 1)[0],
                     'gt': fr['gt'], 'obj': fr['obj'], 'diff': fr['diff'],
                     'distract': fr['distract'], 'pred': pred,
                     'dist': float(((pred[0] - fr['gt'][0]) ** 2 + (pred[1] - fr['gt'][1]) ** 2) ** 0.5),
                     'hit': pck_hit(pred, fr['gt'], fr['obj'], 0.5, 'max'),
                     'hit_min': pck_hit(pred, fr['gt'], fr['obj'], 0.5, 'min')})
    assert not miss_gt, f'{miss_gt} 行挂接 GT 失败（split/实例/帧号不匹配）'
    peaks = None
    if args.peaks:
        peaks = []
        for line in open(args.peaks, encoding='utf-8'):
            r = json.loads(line)
            fr = table[(r['inst'], int(r['f']))]
            peaks.append({'gt': fr['gt'], 'obj': fr['obj'], 'peaks': r['peaks']})
    out = aggregate(recs, peaks)
    out['results_file'] = args.results
    out['split'] = args.split
    txt = json.dumps(out, ensure_ascii=False, indent=1)
    print(txt)
    if args.out:
        Path(args.out).write_text(txt, encoding='utf-8')


if __name__ == '__main__':
    main()
