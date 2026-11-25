# TCIA-v2统一e116b_s42 top20+224峰crop数据管线。train/val/test共用extract_peaks_and_crops。
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import cv2
import numpy as np
import torch

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
os.environ['CSTLOC_PROJ_ADAPTER'] = 'orth'
import tools.n_single_template_eval as n
from data.cstloc_dataset import _letterbox
from tools.cmi_probe_build import build_locator, label_peaks
from tools.tcia_data import (CMI_EXT, OUT, ROOT, TMPL_CACHE, TMPL_SHA, donor_for,
                             load_trials, sha256, train_donor_for, write_jsonl)
from tools.v_verifier_rerank import IMG_SIZE, MIN_DIST, crop_patch, nms_peaks
from tools.wrong_template_probe import load_frame

SEED = 42
K = 20
DEVICE = 'cuda'
OLD_TRAIN_SHA = 'f18f5bbe05d4c52ba89bd38b3045f2cd1ce58f2fb229de7c6abea7cda007ed79'
CKPT = Path(os.environ.get('CSTLOC_LOCATOR_CKPT', str(CODE / 'checkpoints/e116b_s42/best_model.pth')))
PIPELINE_VERSION = 'tcia-v2-peakcrop-v1'


def pipeline_sha():
    h = hashlib.sha256()
    for p in [CODE / 'tools/tcia_v2_data.py', CODE / 'tools/cmi_probe_build.py',
              CODE / 'tools/v_verifier_rerank.py', CODE / 'tools/wrong_template_probe.py']:
        h.update(p.name.encode()); h.update(sha256(p).encode())
    h.update(f'|K={K}|min_dist={MIN_DIST}|IMG={IMG_SIZE}|PATCH=224|{PIPELINE_VERSION}'.encode())
    return h.hexdigest()


def letterboxed_bgr(split, inst, f):
    rgb, _, _, _ = _letterbox(cv2.cvtColor(cv2.imread(str(
        ROOT / split / inst / 'ir/search' / f'{f:06d}.jpg')), cv2.COLOR_BGR2RGB), IMG_SIZE)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def crops_from_peaks(ir_bgr, peaks):
    """所有train/val/test、缓存复用/新提取均只经此224 crop入口。"""
    return [crop_patch(ir_bgr, int(x), int(y)) for x, y, _v in peaks]


@torch.no_grad()
def extract_peaks_and_crops(model, tmpl_entries, fr, ir_bgr, use_id=None):
    """train/val/test唯一现场取峰入口：固定B≤12，与旧CMI/V″生产口径一致。
    输入同帧模板列表，返回逐trial (20峰, 224 crops)；候选crop只经crops_from_peaks。
    use_id=None 时按模型自身 use_id 选择；e134/simple 无专家时不会误走专家分支。"""
    assert 1 <= len(tmpl_entries) <= 12
    if use_id is None:
        use_id = bool(getattr(model, 'use_id', True))
    logits = n.forward_cached(model, use_id, fr['ir'].unsqueeze(0).to(DEVICE), list(tmpl_entries))
    heats = torch.sigmoid(logits)[:, 0].cpu().numpy()
    peaks = [nms_peaks(h, K, MIN_DIST) for h in heats]
    return peaks, [crops_from_peaks(ir_bgr, p) for p in peaks]


def protocol_template_ids(split, inst):
    """N协议val/test枚举与现场取峰共用；test仅在最终阶段调用。"""
    return sorted(int(p.stem) for p in (ROOT / split / inst / 'rgb/template').glob('*.jpg'))


def protocol_frame_trials(split, inst, f, tmpls=None):
    """N协议单帧trial列表：|f−t|≥30；后续val/test只调用同一extract_peaks_and_crops。"""
    tmpls = protocol_template_ids(split, inst) if tmpls is None else tmpls
    return [{'split': split, 'inst': inst, 't': int(t), 'f': int(f)} for t in tmpls if abs(f - t) >= 30]


def has_donor(inst):
    prefix = inst.rsplit('_', 1)[0] + '_'
    return sum(p.is_dir() and p.name.startswith(prefix) and
               any((ROOT / 'train' / p.name / 'rgb/template').glob('*.jpg'))
               for p in (ROOT / 'train').iterdir()) >= 2


def base_record(r, donor_map, origin, source_line=None, pool_index=None, split='train'):
    donor, tmap = donor_map[r['inst']]
    return {'schema': 'tcia_v2.peakcache.v1', 'split': split, 'inst': r['inst'],
            't': r['t'], 'f': r['f'], 'diff': r['diff'], 'gt': r['gt'], 'obj': r['obj'],
            'peaks': r['peaks'], 'labels': r['labels'],
            'donor': {'split': 'train', 'inst': donor, 't': tmap[r['t']]},
            'source': {'origin': origin, 'line': source_line, 'pool_index': pool_index}}


def build_pool(old, allowed_insts):
    pool = []
    by_key = {}
    for r in old:
        by_key.setdefault((r['inst'], r['t']), set()).add(r['f'])
    for inst, t in sorted(by_key):
        assert inst in allowed_insts, f'donor_map/pool实例不一致: {inst}'
        fs = sorted(int(p.stem) for p in (ROOT / 'train' / inst / 'ir/search').glob('*.json'))
        residual = [f for f in fs if abs(f - t) >= 30 and f not in by_key[(inst, t)]]
        for f in residual:
            pool.append((inst, t, f))
    assert len(pool) == 587863, len(pool)
    rng = np.random.RandomState(SEED)
    rng.shuffle(pool)
    return pool


@torch.no_grad()
def verify_reuse(model, old):
    """按原CMI同帧B≤12组成复核10实例：同一函数，坐标全等、峰值<1e-5。"""
    by_inst = {}
    for r in old:
        by_inst.setdefault(r['inst'], []).append(r)
    bad = 0
    for inst in sorted(by_inst)[:10]:
        rs = by_inst[inst]
        f = rs[0]['f']
        chunk = [r for r in rs if r['f'] == f][:12]  # 与原生成顺序一致
        fr = load_frame(ROOT / 'train' / inst, f)
        shard = torch.load(CMI_EXT / f'cmi_probe_tmpl_train_{inst}.pt', map_location='cpu')
        peaks, _ = extract_peaks_and_crops(model, [shard[str(r['t'])] for r in chunk], fr,
                                            letterboxed_bgr('train', inst, f))
        for r, pp in zip(chunk, peaks):
            for i in range(20):
                dc = abs(pp[i][0] - r['peaks'][i][0]) + abs(pp[i][1] - r['peaks'][i][1])
                dv = abs(pp[i][2] - r['peaks'][i][2])
                if dc != 0 or dv >= 1e-5:
                    bad += 1
                    print('REUSE_MISMATCH', inst, r['t'], f, i, 'dc', dc, 'dv', dv, flush=True)
    print(f'[train reuse G0] 10实例同B≤12复算，硬不符={bad}', flush=True)
    assert bad == 0


def phase_build_train():
    assert '4090' in torch.cuda.get_device_name(0)
    assert sha256(TMPL_CACHE) == TMPL_SHA
    free = int(os.popen("free -m | awk 'NR==2{print $7}'").read().strip())
    assert free >= 4000, free
    old_path = OUT / 'cmi_probe_trials_train.jsonl'
    assert sha256(old_path) == OLD_TRAIN_SHA
    old_all = load_trials('train')
    old = [r for r in old_all if r['cover'] and has_donor(r['inst'])]
    assert len(old) == 12031 and len({r['inst'] for r in old}) == 69
    donor_map = {inst: train_donor_for(inst) for inst in sorted({r['inst'] for r in old})}
    reused = [base_record(r, donor_map, 'cmi34_reuse', i + 1) for i, r in enumerate(old)]
    model = build_locator()
    verify_reuse(model, old)
    pool = build_pool([r for r in old_all if r['inst'] in donor_map], set(donor_map))
    supp_path = OUT / 'tcia_v2_train_supplement.jsonl'
    prog_path = OUT / 'tcia_v2_supplement_progress.jsonl'
    existing = [json.loads(l) for l in open(supp_path, encoding='utf-8')] if supp_path.exists() else []
    start_chunk = 0
    if prog_path.exists():
        last = json.loads(open(prog_path, encoding='utf-8').readlines()[-1])
        start_chunk = last['next_chunk']
    assert len(existing) <= 7969
    supplement = list(existing)
    t0 = time.time()
    for ci in range(start_chunk, (len(pool) + 1023) // 1024):
        if len(supplement) >= 7969:
            break
        chunk = [(ci * 1024 + j, *pool[ci * 1024 + j]) for j in range(min(1024, len(pool) - ci * 1024))]
        by_inst = {}
        for item in chunk:
            by_inst.setdefault(item[1], []).append(item)
        found = []
        for inst, items in sorted(by_inst.items()):
            tmpl = torch.load(CMI_EXT / f'cmi_probe_tmpl_train_{inst}.pt', map_location='cpu')
            by_frame = {}
            for item in items:
                by_frame.setdefault(item[3], []).append(item)
            for f, fitems in sorted(by_frame.items()):
                fr = load_frame(ROOT / 'train' / inst, f)
                ir_bgr = letterboxed_bgr('train', inst, f)
                for s in range(0, len(fitems), 12):
                    sub = fitems[s:s + 12]
                    feats = [tmpl[str(it[2])] for it in sub]
                    peak_list, _ = extract_peaks_and_crops(model, feats, fr, ir_bgr)
                    for it, peaks in zip(sub, peak_list):
                        labels = label_peaks(peaks, fr['gt_box'], fr['distractor_boxes'])
                        if 2 not in labels:
                            continue
                        rr = {'inst': inst, 't': it[2], 'f': f, 'diff': fr['difficulty'],
                              'gt': [round(float(v), 2) for v in fr['gt_center']],
                              'obj': [round(float(v), 2) for v in fr['obj_size']],
                              'peaks': [[int(x), int(y), float(v)] for x, y, v in peaks], 'labels': labels}
                        found.append((it[0], base_record(rr, donor_map, 'seed42_supplement', pool_index=it[0])))
        found.sort(key=lambda x: x[0])
        for _idx, rec in found:
            if len(supplement) < 7969:
                supplement.append(rec)
        write_jsonl(supp_path, supplement)
        write_jsonl(prog_path, [{'next_chunk': ci + 1, 'covered': len(supplement),
                                'elapsed_s': time.time() - t0}])
        print(f'[train supplement] chunk={ci} covered={len(supplement)}/7969 elapsed={time.time()-t0:.0f}s', flush=True)
    assert len(supplement) == 7969, len(supplement)
    all_rows = reused + supplement
    all_rows.sort(key=lambda r: (r['inst'], r['t'], r['f']))
    assert len(all_rows) == 20000
    assert len({(r['inst'], r['t'], r['f']) for r in all_rows}) == 20000
    assert all(2 in r['labels'] and len(r['peaks']) == 20 and r['donor']['inst'] != r['inst'] for r in all_rows)
    train_path = OUT / 'tcia_v2_train_trials.jsonl'
    write_jsonl(train_path, all_rows)
    manifest = [{'type': 'train_manifest', 'n_trials': len(all_rows), 'n_reused': len(reused),
                 'n_supplement': len(supplement), 'instances': len({r['inst'] for r in all_rows}),
                 'seed': SEED, 'algorithm': 'tcia-v2-supplement-v1', 'pipeline_sha256': pipeline_sha(),
                 'train_sha256': sha256(train_path), 'supplement_sha256': sha256(supp_path),
                 'cmi_train_sha256': OLD_TRAIN_SHA, 'n_cache_tmpl_sha256': sha256(TMPL_CACHE),
                 'e116b_ckpt_sha256': sha256(CKPT), 'extract_fn': 'extract_peaks_and_crops/crops_from_peaks'}]
    write_jsonl(OUT / 'tcia_v2_train_manifest.jsonl', manifest)
    print('TRAIN_DATA DONE', manifest[0], flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', required=True, choices=['build_train'])
    a = ap.parse_args()
    {'build_train': phase_build_train}[a.phase]()


if __name__ == '__main__':
    main()
