# 任务书3.6 TCIA：只读数据/ donor / 指标公共函数。训练采样在后续阶段加入。
import hashlib
import json
import os
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
OUT = CODE / 'diag_out'
CACHE = OUT / 'tcia_cache'
CKPT_DIR = CODE / 'checkpoints' / 'tcia_lora'
CMI_EXT = Path(os.environ.get('CSTLOC_CMI_CACHE', '/home/user/Code/cmi_probe_cache'))
ROOT = Path(os.environ.get('CSTLOC_DATA_ROOT', '/home/user/Code/Datasets/CSTLoc'))
TMPL_CACHE = OUT / 'n_cache_tmpl.pt'
TMPL_SHA = '79327010399d583ec48a19952b59513b9d30ab78d0ad58e63412a3f2f3c50334'
M0_INSTS = ('bus_030', 'car_088', 'car_095', 'pedestrian_133', 'tricycle_018', 'car_117')
UNSEEN = ('animal_', 'truck_')
SEED = 42


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def write_jsonl(path, rows, mode='w'):
    with open(path, mode, encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f'WRITE {path.name}: {len(rows)} rows sha256={sha256(path)}', flush=True)


def load_trials(split):
    return [json.loads(l) for l in open(OUT / f'cmi_probe_trials_{split}.jsonl', encoding='utf-8')]


def template_ids(split, inst):
    return sorted(int(p.stem) for p in (ROOT / split / inst / 'rgb/template').glob('*.jpg'))


def donor_for(inst):
    """train同类字典序首实例；wrong永不是目标实例。donor_t=同t，缺失取最近（并列小）。"""
    prefix = inst.rsplit('_', 1)[0] + '_'
    donors = sorted(p.name for p in (ROOT / 'train').iterdir()
                    if p.is_dir() and p.name.startswith(prefix) and template_ids('train', p.name))
    assert donors, inst
    donor = donors[0]
    assert donor != inst
    ts = template_ids('train', donor)
    return donor, {t: min(ts, key=lambda u: (abs(u - t), u)) for t in template_ids('val', inst)}


def metric_pair_top1(labels, scores):
    ps = [i for i, v in enumerate(labels) if v == 2]
    hs = [i for i, v in enumerate(labels) if v == 1]
    pair = top1 = margin = None
    if ps:
        top1 = float(labels[int(scores.argmax())] == 2)
    if ps and hs:
        vals = []
        for i in ps:
            for j in hs:
                d = float(scores[i] - scores[j])
                vals.append(1.0 if d > 0 else 0.5 if d == 0 else 0.0)
        pair = float(sum(vals) / len(vals))
        margin = float(scores[ps].max() - scores[hs].max())
    return pair, top1, margin

# ---------------- M1 train trials / 图像加载 ----------------
import argparse
import cv2
import numpy as np
import torch
from tools.v_verifier_rerank import norm224


def frame_ids(split, inst):
    return sorted(int(p.stem) for p in (ROOT / split / inst / 'ir/search').glob('*.json'))


def train_donor_for(inst):
    """train目标：同类train实例字典序循环下一实例，保证非目标。"""
    prefix = inst.rsplit('_', 1)[0] + '_'
    peers = sorted(p.name for p in (ROOT / 'train').iterdir()
                   if p.is_dir() and p.name.startswith(prefix) and template_ids('train', p.name))
    i = peers.index(inst)
    donor = peers[(i + 1) % len(peers)]
    assert donor != inst
    ts = template_ids('train', donor)
    return donor, {t: min(ts, key=lambda u: (abs(u - t), u)) for t in template_ids('train', inst)}


def load_meta(split, inst, f):
    return json.loads((ROOT / split / inst / 'ir/search' / f'{f:06d}.json').read_text(encoding='utf-8'))


def shape_box(shape):
    (x1, y1), (x2, y2) = shape['points']
    return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]


def jitter_box(box, which):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    dx, dy, scale = (-0.10, 0.10, 0.90) if which == 0 else (0.10, -0.10, 1.10)
    cx, cy = (x1 + x2) / 2 + dx * w, (y1 + y2) / 2 + dy * h
    nw, nh = w * scale, h * scale
    return [cx - nw / 2, cy - nh / 2, cx + nw / 2, cy + nh / 2]


def build_m1_trials():
    rng = np.random.RandomState(SEED)
    all_insts = sorted(p.name for p in (ROOT / 'train').iterdir() if p.is_dir())
    def has_donor(inst):
        prefix = inst.rsplit('_', 1)[0] + '_'
        return sum(p.is_dir() and p.name.startswith(prefix) and bool(template_ids('train', p.name))
                   for p in (ROOT / 'train').iterdir()) >= 2
    skipped = [i for i in all_insts if not has_donor(i)]
    keys = [(inst, t) for inst in all_insts if has_donor(inst) for t in template_ids('train', inst)]
    print('[M1 data] train单例类无合法wrong donor，跳过实例:', skipped, 'eligible_templates=', len(keys), flush=True)
    assert len(keys) > 4000, len(keys)
    extra_idx = set(rng.choice(len(keys), 20000 - 4 * len(keys), replace=False).tolist())
    rows = []
    donor_cache = {}
    for ki, (inst, t) in enumerate(keys):
        k = 5 if ki in extra_idx else 4
        fs = [f for f in frame_ids('train', inst) if abs(f - t) >= 30]
        assert len(fs) >= k
        take = rng.choice(len(fs), k, replace=False)
        if inst not in donor_cache:
            donor_cache[inst] = train_donor_for(inst)
        donor, tmap = donor_cache[inst]
        donor_frames = frame_ids('train', donor)
        for f in sorted(fs[int(i)] for i in take):
            meta = load_meta('train', inst, f)
            target = shape_box(meta['shapes'][0])
            distractors = [shape_box(s) for s in meta['shapes'][1:] if not s['flags'].get('target')]
            cand = [{'split': 'train', 'inst': inst, 'f': int(f), 'box': target,
                     'kind': 'gt', 'label': 2}]
            if distractors:
                nd = min(2, len(distractors))
                for di in rng.choice(len(distractors), nd, replace=False):
                    cand.append({'split': 'train', 'inst': inst, 'f': int(f),
                                 'box': distractors[int(di)], 'kind': 'same_scene_distractor', 'label': 1})
            # 其他实例随机框：来自wrong donor的另一随机帧/随机shape；不是目标实例，标0（非same-scene难负）
            of = int(donor_frames[int(rng.randint(len(donor_frames)))])
            om = load_meta('train', donor, of)
            oi = int(rng.randint(len(om['shapes'])))
            cand.append({'split': 'train', 'inst': donor, 'f': of, 'box': shape_box(om['shapes'][oi]),
                         'kind': 'other_instance_random', 'label': 0})
            for j in range(2):
                cand.append({'split': 'train', 'inst': inst, 'f': int(f), 'box': jitter_box(target, j),
                             'kind': f'gt_jitter{j}', 'label': 2})
            rows.append({'type': 'm1_train_trial', 'split': 'train', 'inst': inst, 't': t, 'f': int(f),
                         'wrong_split': 'train', 'wrong_inst': donor, 'wrong_t': tmap[t],
                         'candidates': cand})
    assert len(rows) == 20000, len(rows)
    assert all(r['wrong_inst'] != r['inst'] for r in rows)
    write_jsonl(OUT / 'tcia_train_trials.jsonl', rows)
    hist = {}
    for r in rows:
        for c in r['candidates']:
            hist[c['kind']] = hist.get(c['kind'], 0) + 1
    print('[M1 data] trials=20000 candidate_hist=', hist, flush=True)


def crop_with_margin(img, box, margin=1.2):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    nw, nh = (x2 - x1) * margin, (y2 - y1) * margin
    xa = max(0, int(round(cx - nw / 2))); xb = min(w, int(round(cx + nw / 2)))
    ya = max(0, int(round(cy - nh / 2))); yb = min(h, int(round(cy + nh / 2)))
    crop = img[ya:yb, xa:xb]
    if crop.size == 0:  # 越界框抖动时回退到最近1px，确定性且不丢候选
        xc = min(w - 1, max(0, int(round(cx)))); yc = min(h - 1, max(0, int(round(cy))))
        crop = img[yc:yc + 1, xc:xc + 1]
    return cv2.resize(crop, (224, 224), interpolation=cv2.INTER_LINEAR)


def load_template_tensor(split, inst, t):
    im = cv2.cvtColor(cv2.imread(str(ROOT / split / inst / 'rgb/template' / f'{t:06d}.jpg')), cv2.COLOR_BGR2RGB)
    assert im is not None
    return norm224(cv2.resize(im, (224, 224), interpolation=cv2.INTER_LINEAR))


def load_trial_images(r):
    real = load_template_tensor('train', r['inst'], r['t'])
    wrong = load_template_tensor(r['wrong_split'], r['wrong_inst'], r['wrong_t'])
    cache = {}
    crops = []
    for c in r['candidates']:
        key = (c['split'], c['inst'], c['f'])
        if key not in cache:
            sp, inst, f = key
            im = cv2.cvtColor(cv2.imread(str(ROOT / sp / inst / 'ir/search' / f'{f:06d}.jpg')),
                              cv2.COLOR_BGR2RGB)
            assert im is not None
            cache[key] = im
        crops.append(norm224(crop_with_margin(cache[key], c['box'])))
    return real, wrong, torch.stack(crops), torch.tensor([c['label'] for c in r['candidates']], dtype=torch.long)


def collate_m1(rows):
    dat = [load_trial_images(r) for r in rows]
    # 仅批处理性能：每trial零填充到6候选，真实labels/lengths不变，loss只读前lengths个。
    # 固定shape避免bf16 fused-attention在每个新候选总数上反复触发慢路径。
    cx = []
    for d in dat:
        x = d[2]
        if x.shape[0] < 6:
            x = torch.cat([x, torch.zeros(6 - x.shape[0], *x.shape[1:], dtype=x.dtype)], dim=0)
        cx.append(x)
    return {'real': torch.stack([d[0] for d in dat]), 'wrong': torch.stack([d[1] for d in dat]),
            'cand': torch.cat(cx), 'labels': [d[3] for d in dat],
            'lengths': [len(d[3]) for d in dat]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', required=True, choices=['build_m1'])
    a = ap.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    if a.phase == 'build_m1':
        build_m1_trials()


if __name__ == '__main__':
    main()
