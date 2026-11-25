#!/usr/bin/env python3
# TCIA-v2 3.8 A版：完整 val/test 的 TCIA-单独选择器。
# 与 3.6b 完全相同的 top-20峰+224crop 管线；不融合 V″；不乘 α。
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

from models.tcia_v2 import TCIAv2
from tools.tcia_data import (CMI_EXT, OUT, ROOT, load_template_tensor, sha256,
                             write_jsonl)
from tools.tcia_v2_data import (crops_from_peaks, extract_peaks_and_crops,
                                letterboxed_bgr, pipeline_sha,
                                protocol_frame_trials, protocol_template_ids)
from tools.cmi_probe_build import build_locator
from tools.v_verifier_rerank import norm224
from tools.wrong_template_probe import load_frame

DEVICE = 'cuda'
SEED = int(os.environ.get('TCIA_SEED', '42'))
assert SEED in (42, 123, 2024)
VAL_TOTAL = int(os.environ.get('TCIA_VAL_TOTAL', '171587'))
TEST_TOTAL = int(os.environ.get('TCIA_TEST_TOTAL', '327772'))
UNSEEN_PREFIX = ('animal_', 'truck_')


def select_path(name):
    return OUT / f'tcia_v2_s{SEED}_{name}.jsonl'


def score_dir_for(split):
    d = OUT / f'tcia_v2_s{SEED}_{split}_tcia_scores'
    d.mkdir(parents=True, exist_ok=True)
    return d


def ckpt_dir_for_seed(seed):
    return CODE / ('checkpoints/tcia_v2' if seed == 42 else f'checkpoints/tcia_v2_s{seed}')


def best_fullval_path(seed):
    return Path(os.environ.get('TCIA_BEST_CKPT', str(ckpt_dir_for_seed(seed) / 'best_fullval.pt')))


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
    assert '4090' in torch.cuda.get_device_name(0)
    free = int(os.popen("free -m | awk 'NR==2{print $7}'").read().strip())
    assert free >= 4000, free


@torch.no_grad()
def run_split(model, split, instances=None, write_scores=True, staging_dir=None):
    """TCIA-单独 pass。model: TCIAv2（eval 模式，已在 DEVICE）。
    返回 (summary, rows)。rows 逐 trial 含 hit_tcia / hit_locator。"""
    assert split in ('val', 'test')
    _assert_env()
    locator = build_locator()
    expected_total = VAL_TOTAL if split == 'val' else TEST_TOTAL
    test_cache = torch.load(OUT / 'n_cache_tmpl_test.pt', map_location='cpu') if split == 'test' else None
    all_insts = sorted(p.name for p in (ROOT / split).iterdir()
                       if p.is_dir() and protocol_template_ids(split, p.name))
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
        tmpls = protocol_template_ids(split, inst)
        frames = sorted(int(p.stem) for p in (ROOT / split / inst / 'ir/search').glob('*.json'))
        expected_inst = sum(len(protocol_frame_trials(split, inst, f, tmpls)) for f in frames)
        cache = test_cache[inst] if split == 'test' else torch.load(
            CMI_EXT / f'cmi_probe_tmpl_val_{inst}.pt', map_location='cpu')
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
            store = None
            if staging_dir is not None:
                import importlib.util as _ilu
                _cspec = _ilu.spec_from_file_location('crop_staging', Path('/home/user/Code/CSTLoc/evaluation/cache/v3_test_diag/crop_staging.py'))
                _csm = _ilu.module_from_spec(_cspec); _cspec.loader.exec_module(_csm)
                store = _csm.CropStore(tag=f'{inst}_{f}')
            for r, peaks, crops in trial_data:
                for (x, y, _v), crop in zip(peaks, crops):
                    key = (int(x), int(y))
                    if key not in seen:
                        seen[key] = len(uniq)
                        uniq.append((key, store.put(crop) if store is not None else crop))
            zs = []
            for s in range(0, len(uniq), 128):
                batch = [store.get(c) if store is not None else c for _k, c in uniq[s:s + 128]]
                x = torch.stack([norm224(c) for c in batch]).to(DEVICE)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    zs.append(model.encode_head(x, 'cand').float())
                del batch
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
            if store is not None:
                store.close()
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
               'pipeline_sha256': pipeline_sha(), 'elapsed_s': time.time() - t0, **flat,
               'detail': metrics}
    return summary, rows


def merge_select_rows(existing, new_rows):
    rows = [r for r in existing if r.get('type') != 'best_selected']
    seen = {r['epoch'] for r in rows}
    for r in new_rows:
        if r.get('epoch') not in seen:
            rows.append(r)
            seen.add(r['epoch'])
    return sorted(rows, key=lambda r: r['epoch'])


def load_adapter(ck):
    state = torch.load(ck, map_location='cpu')
    model = TCIAv2().to(DEVICE).eval()
    model.load_adapter_state_dict(state['state'])
    return model, state


def select(ckpts=None, split='val', resume=False):
    """逐 ckpt 完整 val（TCIA-单独），按 overall 选 best（tie hard/seen），存 best_fullval.pt。
    resume=True 时读取已有 fullval_select 行，跳过已完成 epoch 继续。"""
    assert split == 'val'
    cdir = ckpt_dir_for_seed(SEED)
    log = select_path('fullval_select')
    existing = [json.loads(l) for l in open(log, encoding='utf-8')] if (resume and log.exists()) else []
    done = {r['epoch'] for r in existing if r.get('type') != 'best_selected'}
    if ckpts is None:
        ckpts = sorted(cdir.glob('epoch_*.pt'))
    todo = [ck for ck in ckpts if int(ck.stem.split('_')[1]) not in done]
    assert todo, (cdir, done)
    results = [r for r in existing if r.get('type') != 'best_selected']
    for ck in todo:
        model, state = load_adapter(ck)
        summary, _rows = run_split(model, 'val', write_scores=False)
        summary.update({'epoch': state.get('epoch'), 'ckpt': ck.name, 'ckpt_sha256': sha256(ck)})
        results = merge_select_rows(results, [summary])
        write_jsonl(select_path('fullval_select'), results)
        print('CKPT_EVAL', json.dumps({k: summary[k] for k in
                  ('epoch', 'hit_tcia_overall', 'hit_tcia_hard', 'hit_tcia_seen', 'hit_tcia_unseen',
                   'n_trials', 'elapsed_s')}, ensure_ascii=False), flush=True)
        del model
        torch.cuda.empty_cache()
    best = max(results, key=lambda r: (r['hit_tcia_overall'], r['hit_tcia_hard'], r['hit_tcia_seen']))
    best_row = {'type': 'best_selected', 'seed': SEED, 'epoch': best['epoch'],
                'ckpt_sha256': best['ckpt_sha256'], 'hit_tcia_overall': best['hit_tcia_overall'],
                'rule': 'max hit_tcia_overall, tie hard, seen; full val 197649 trials; TCIA only'}
    write_jsonl(select_path('fullval_select'), results + [best_row])
    src = cdir / f"epoch_{best['epoch']:02d}.pt"
    dst = best_fullval_path(SEED)
    state = torch.load(src, map_location='cpu')
    torch.save({'epoch': best['epoch'], 'state': state['state'],
                'selected': best_row, 'model_version': TCIAv2.MODEL_VERSION,
                'pipeline_sha256': pipeline_sha()}, dst)
    print('BEST_FULLVAL', json.dumps(best_row, ensure_ascii=False),
          f'sha={sha256(dst)}', flush=True)
    return best_row


def final_test(ck=None, staging_dir=None):
    if ck is None:
        ck = best_fullval_path(SEED)
    ck = Path(ck)
    assert ck.exists(), ck
    assert '4090' in torch.cuda.get_device_name(0), torch.cuda.get_device_name(0)
    model, _state = load_adapter(ck)
    summary, _rows = run_split(model, 'test', staging_dir=staging_dir)
    summary['type'] = 'test_tcia_only'
    summary['best_fullval_sha256'] = sha256(ck)
    write_jsonl(select_path('test_summary'), [summary])
    print('TEST_TCIA_ONLY', json.dumps({k: summary[k] for k in
          ('hit_tcia_overall', 'hit_tcia_hard', 'hit_tcia_seen', 'hit_tcia_unseen',
           'hit_locator_overall', 'n_trials', 'elapsed_s')}, ensure_ascii=False), flush=True)
    return summary


def smoke():
    """seed42 epoch_00 两个实例冒烟；并用 epoch_01 与旧 val scores 交叉验证管线一致。"""
    _assert_env()
    cdir = ckpt_dir_for_seed(SEED)
    insts = ['animal_003', 'bus_030']
    model, _ = load_adapter(cdir / 'epoch_00.pt')
    summary, rows = run_split(model, 'val', instances=insts)
    print('SMOKE_EPOCH00', json.dumps({k: summary[k] for k in
          ('hit_tcia_overall', 'hit_tcia_hard', 'hit_tcia_unseen', 'n_trials')}, ensure_ascii=False), flush=True)
    del model
    torch.cuda.empty_cache()
    # 管线交叉验证：epoch_01（旧 M0 best）与 3.6b 落盘 val scores 的 argmax(tcia) 逐 trial 一致
    model, _ = load_adapter(cdir / 'epoch_01.pt')
    summary1, rows1 = run_split(model, 'val', instances=insts, write_scores=False)
    del model
    torch.cuda.empty_cache()
    stored = {}
    for inst in insts:
        for ln in open(OUT / f'tcia_v2_val_scores_{inst}.jsonl', encoding='utf-8'):
            r = json.loads(ln)
            stored[(r['inst'], r['t'], r['f'])] = r
    agree = total = 0
    for r in rows1:
        s = stored[(r['inst'], r['t'], r['f'])]
        gt, obj = s['gt'], s['obj']
        pi = int(np.argmax(np.asarray(s['tcia'])))
        hit_stored = bool(np.hypot(s['peaks'][pi][0] - gt[0], s['peaks'][pi][1] - gt[1]) <= 0.5 * max(obj))
        total += 1
        agree += int(hit_stored == r['hit_tcia'])
    row = {'type': 'smoke_pipeline_check', 'seed': SEED, 'n_compared': total,
           'tcia_argmax_hit_agree': agree,
           'pass': bool(total > 0 and agree == total)}
    write_jsonl(select_path('smoke'), [row])
    print('SMOKE_PIPELINE_CHECK', json.dumps(row, ensure_ascii=False), flush=True)
    assert row['pass']


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
    assert select_path('fullval_select').name == f'tcia_v2_s{SEED}_fullval_select.jsonl'
    existing = [{'type': 'ckpt', 'epoch': 0, 'hit_tcia_overall': 0.5},
                {'type': 'ckpt', 'epoch': 1, 'hit_tcia_overall': 0.4},
                {'type': 'best_selected', 'epoch': 0}]
    new_rows = [{'type': 'ckpt', 'epoch': 1, 'hit_tcia_overall': 0.45},
                {'type': 'ckpt', 'epoch': 2, 'hit_tcia_overall': 0.6}]
    merged = merge_select_rows(existing, new_rows)
    assert [r['epoch'] for r in merged] == [0, 1, 2], merged
    assert merged[1]['hit_tcia_overall'] == 0.4, '旧行优先，不覆盖'
    assert all(r.get('type') != 'best_selected' for r in merged)
    print('MERGE_OK', len(merged))
    print('SELECT_SELF_TEST_PASS', SEED)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', required=True, choices=['self_test', 'smoke', 'select', 'test'])
    ap.add_argument('--seed', type=int, default=None, choices=[42, 123, 2024])
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--ckpt', default=None)
    ap.add_argument('--staging-dir', default=None, help='候选 crop 存盘目录（disk-backed crop staging，内存安全）')
    args = ap.parse_args()
    if args.seed is not None:
        global SEED
        SEED = args.seed
        os.environ['TCIA_SEED'] = str(SEED)
    if args.phase == 'self_test':
        self_test()
    elif args.phase == 'smoke':
        smoke()
    elif args.phase == 'select':
        select(resume=args.resume)
    elif args.phase == 'test':
        final_test(ck=args.ckpt, staging_dir=args.staging_dir)


if __name__ == '__main__':
    main()