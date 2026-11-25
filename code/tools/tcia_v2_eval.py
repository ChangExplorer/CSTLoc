# TCIA-v2 G0 / M0 α选择 / full val / 单次full test。
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import cv2
import numpy as np
import torch
import torch.nn.functional as F

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
from models.encoders.dinov2_encoder import DINOv2TemplateEncoder
from models.tcia_v2 import TCIAv2
from tools.tcia_data import OUT, ROOT, TMPL_CACHE, TMPL_SHA, load_template_tensor, sha256, write_jsonl
from tools.tcia_v2_data import crops_from_peaks, letterboxed_bgr
from tools.v_verifier_rerank import norm224

DEVICE = 'cuda'
SEED = int(os.environ.get('TCIA_SEED', '42'))
assert SEED in (42, 123, 2024), SEED
ALPHAS = (0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0)


def artifact(name):
    return OUT / (f'tcia_v2_{name}.jsonl' if SEED == 42 else f'tcia_v2_s{SEED}_{name}.jsonl')


def artifact_glob():
    return 'tcia_v2_*.jsonl' if SEED == 42 else f'tcia_v2_s{SEED}_*.jsonl'


def score_glob(split):
    return (f'tcia_v2_{split}_scores_*.jsonl' if SEED == 42
            else f'tcia_v2_s{SEED}_{split}_scores_*.jsonl')


def ckpt_path():
    return CODE / ('checkpoints/tcia_v2/best.pt' if SEED == 42
                   else f'checkpoints/tcia_v2_s{SEED}/best.pt')


def test_completed():
    return artifact('test_complete').exists()


def zscore(x):
    x = np.asarray(x, dtype=np.float64)
    sd = x.std()
    return (x - x.mean()) / sd if sd > 0 else np.zeros_like(x)


def fuse(vpp, tcia, alpha):
    return zscore(vpp) + float(alpha) * zscore(tcia)


@torch.no_grad()
def phase_g0():
    assert '4090' in torch.cuda.get_device_name(0)
    assert sha256(TMPL_CACHE) == TMPL_SHA
    model = TCIAv2().to(DEVICE).eval()
    ref_model = DINOv2TemplateEncoder(pretrained=True).to(DEVICE).eval()
    records = [json.loads(l) for l in open(OUT / 'tcia_m0_trials.jsonl', encoding='utf-8')][:10]
    src_map = {(r['inst'], r['t'], r['f']): r for r in
               [json.loads(l) for l in open(OUT / 'cmi_probe_trials_val.jsonl', encoding='utf-8')]}
    diffs = []
    for r in records:
        real = load_template_tensor('val', r['inst'], r['t']).unsqueeze(0).to(DEVICE)
        src = src_map[(r['inst'], r['t'], r['f'])]
        ir_bgr = letterboxed_bgr('val', r['inst'], r['f'])
        cx = torch.stack([norm224(c) for c in crops_from_peaks(ir_bgr, src['peaks'])]).to(DEVICE)
        zt, zc = model.scores(real, cx, lora_enabled=False)
        got = (zc @ zt[0]).cpu().numpy()
        rt = ref_model.backbone.get_intermediate_layers(real, n=[15, 23], reshape=False)
        rc = ref_model.backbone.get_intermediate_layers(cx, n=[15, 23], reshape=False)
        rtg = F.normalize(torch.cat([v.mean(1) for v in rt], 1), dim=1)
        rcg = F.normalize(torch.cat([v.mean(1) for v in rc], 1), dim=1)
        ref = (rcg @ rtg[0]).cpu().numpy()
        diffs.append(float(np.max(np.abs(got - ref))))
    alpha_diffs, pick_bad = [], 0
    for r in records:
        v = np.asarray(r['scores']['v2_real'], dtype=np.float64)
        t = np.asarray(r['scores']['raw_real'], dtype=np.float64)
        fused = fuse(v, t, 0)
        alpha_diffs.append(float(np.max(np.abs(fused - zscore(v)))))
        pick_bad += int(np.argmax(fused) != np.argmax(v))
    ok = max(diffs) == 0 and max(alpha_diffs) < 1e-12 and pick_bad == 0
    row = {'type': 'g0', 'raw_max_abs_diff': max(diffs), 'alpha0_max_abs_z_diff': max(alpha_diffs),
           'alpha0_pick_mismatch': pick_bad, 'n_trials': len(records), 'pass': ok}
    write_jsonl(artifact('g0'), [row])
    print('G0', row, flush=True)
    assert ok


@torch.no_grad()
def compute_m0_scores(model, out_path=None):
    records = [json.loads(l) for l in open(OUT / 'tcia_m0_trials.jsonl', encoding='utf-8')]
    src_map = {(r['inst'], r['t'], r['f']): r for r in
               [json.loads(l) for l in open(OUT / 'cmi_probe_trials_val.jsonl', encoding='utf-8')]}
    rows = []
    return _compute_m0_scores_impl(model, records, src_map, out_path)


def m0_scores():
    ck = ckpt_path()
    assert ck.exists(), ck
    state = torch.load(ck, map_location='cpu')
    assert state['data_sha256'] == '4e4b27a0a41e0b9b659b12a1a5327facf87fc6de62759f98eacba3b8602a5b43'
    from tools.tcia_v2_data import pipeline_sha
    assert state['pipeline_sha256'] == pipeline_sha(), '训练/评测候选管线sha不一致'
    model = TCIAv2().to(DEVICE).eval(); model.load_adapter_state_dict(state['state'])
    return _compute_m0_scores_impl(model, records=None, src_map=None,
                                   out_path=artifact('m0_scores'))


@torch.no_grad()
def _compute_m0_scores_impl(model, records, src_map, out_path):
    if records is None:
        records = [json.loads(l) for l in open(OUT / 'tcia_m0_trials.jsonl', encoding='utf-8')]
        src_map = {(r['inst'], r['t'], r['f']): r for r in
                   [json.loads(l) for l in open(OUT / 'cmi_probe_trials_val.jsonl', encoding='utf-8')]}
    rows = []
    for i, r in enumerate(records):
        real = load_template_tensor('val', r['inst'], r['t'])
        wrong = load_template_tensor(r['donor_split'], r['donor_inst'], r['donor_t'])
        tx = torch.stack([real, wrong]).to(DEVICE)
        zt = model.encode_head(tx, 'tmpl')
        ir_bgr = letterboxed_bgr('val', r['inst'], r['f'])
        src = src_map[(r['inst'], r['t'], r['f'])]
        cx = torch.stack([norm224(c) for c in crops_from_peaks(ir_bgr, src['peaks'])]).to(DEVICE)
        zc = model.encode_head(cx, 'cand')
        sr, sw = (zc @ zt[0]).cpu().numpy(), (zc @ zt[1]).cpu().numpy()
        rows.append({'inst': r['inst'], 't': r['t'], 'f': r['f'], 'labels': r['labels'],
                     'donor_inst': r['donor_inst'], 'donor_t': r['donor_t'],
                     'v2_real': r['scores']['v2_real'], 'v2_wrong': r['scores']['v2_wrong'],
                     'tcia_real': [float(v) for v in sr], 'tcia_wrong': [float(v) for v in sw]})
        if (i + 1) % 200 == 0:
            print(f'[M0 scores] {i+1}/{len(records)}', flush=True)
    if out_path is not None:
        write_jsonl(out_path, rows)
    return rows


def evaluate_m0_for_selection(model, epoch):
    rows = compute_m0_scores(model)
    real = candidate_metrics(rows, lambda r: fuse(r['v2_real'], r['tcia_real'], 0.3))
    wrong = candidate_metrics(rows, lambda r: fuse(r['v2_wrong'], r['tcia_wrong'], 0.3))
    tcia_real = candidate_metrics(rows, lambda r: r['tcia_real'])
    return {'type': 'val_epoch', 'epoch': epoch, 'alpha_for_selection': 0.3,
            'fusion_top1': real['top1'], 'fusion_pair': real['pair'],
            'real_tcia_top1': tcia_real['top1'], 'wrong_fusion_top1': wrong['top1'],
            'gap_real_minus_wrong_pp': (real['top1'] - wrong['top1']) * 100}


def candidate_metrics(rows, score_key):
    pairs, tops = [], []
    for r in rows:
        lab = np.asarray(r['labels']); sc = np.asarray(score_key(r), dtype=np.float64)
        ps, hs = np.where(lab == 2)[0], np.where(lab == 1)[0]
        if len(ps):
            tops.append(float(lab[int(sc.argmax())] == 2))
        if len(ps) and len(hs):
            pairs.append(float(np.mean([1.0 if sc[i] > sc[j] else 0.5 if sc[i] == sc[j] else 0.0
                                        for i in ps for j in hs])))
    return {'top1': float(np.mean(tops)), 'n_top1': len(tops),
            'pair': float(np.mean(pairs)), 'n_pair': len(pairs)}


def phase_m0_alpha():
    rows = m0_scores()
    results = []
    for a in ALPHAS:
        real = candidate_metrics(rows, lambda r, a=a: fuse(r['v2_real'], r['tcia_real'], a))
        wrong = candidate_metrics(rows, lambda r, a=a: fuse(r['v2_wrong'], r['tcia_wrong'], a))
        row = {'type': 'alpha', 'alpha': a, 'real': real, 'wrong': wrong,
               'gap_real_minus_wrong_pp': (real['top1'] - wrong['top1']) * 100}
        results.append(row)
        print('ALPHA', json.dumps(row, ensure_ascii=False), flush=True)
    # 预登记：real top1最大；tie依次real pair、real-wrong gap、较小α
    best = max(results, key=lambda r: (r['real']['top1'], r['real']['pair'],
                                       r['gap_real_minus_wrong_pp'], -r['alpha']))
    out = {'type': 'alpha_selected', 'alpha': best['alpha'], 'rule': 'max real top1,tie real pair,gap,smaller alpha',
           'result': best, 'epoch': torch.load(ckpt_path(), map_location='cpu')['epoch']}
    write_jsonl(artifact('alpha'), results + [out])
    print('ALPHA_SELECTED', json.dumps(out, ensure_ascii=False), flush=True)

@torch.no_grad()
def full_pass(split, alpha):
    from models.identity_expert import IdentityExpert
    from tools.cmi_probe_build import build_locator, label_peaks
    from tools.tcia_v2_data import (CMI_EXT, extract_peaks_and_crops, protocol_frame_trials,
                                    protocol_template_ids)
    from tools.v3_verifier_deepen import frame_minmax
    from tools.v_verifier_rerank import EXPERT_CKPT
    from tools.wrong_template_probe import load_frame
    assert split in ('val', 'test')
    complete_marker = artifact(f'{split}_complete')
    if complete_marker.exists():
        raise RuntimeError(f'{split}已完成，禁止重复读取原始{split}')
    assert '4090' in torch.cuda.get_device_name(0)
    free = int(os.popen("free -m | awk 'NR==2{print $7}'").read().strip())
    assert free >= 4000, free
    ck = ckpt_path()
    state = torch.load(ck, map_location='cpu')
    from tools.tcia_v2_data import pipeline_sha
    assert state['pipeline_sha256'] == pipeline_sha(), '训练/评测候选管线sha不一致'
    tcia = TCIAv2().to(DEVICE).eval(); tcia.load_adapter_state_dict(state['state'])
    locator = build_locator()
    expert = IdentityExpert(str(EXPERT_CKPT)).to(DEVICE).eval()
    test_cache = torch.load(OUT / 'n_cache_tmpl_test.pt', map_location='cpu') if split == 'test' else None
    expected_total = 197649 if split == 'val' else 307527
    instances = sorted(p.name for p in (ROOT / split).iterdir()
                       if p.is_dir() and protocol_template_ids(split, p.name))
    total_done, t0 = 0, time.time()
    for inst in instances:
        tmpls = protocol_template_ids(split, inst)
        frames = sorted(int(p.stem) for p in (ROOT / split / inst / 'ir/search').glob('*.json'))
        expected_inst = sum(len(protocol_frame_trials(split, inst, f, tmpls)) for f in frames)
        out_path = artifact(f'{split}_scores_{inst}')
        done_keys = set()
        if out_path.exists():
            for ln in open(out_path, encoding='utf-8'):
                r = json.loads(ln); done_keys.add((r['t'], r['f']))
        if len(done_keys) == expected_inst:
            total_done += len(done_keys)
            print(f'[{split} {inst}] 已完成{len(done_keys)}，跳过原始帧', flush=True)
            continue
        cache = test_cache[inst] if split == 'test' else torch.load(
            CMI_EXT / f'cmi_probe_tmpl_val_{inst}.pt', map_location='cpu')
        # 模板TCIA与V″专家向量每实例只算一次
        tz, egap = {}, {}
        for s in range(0, len(tmpls), 32):
            ts = tmpls[s:s + 32]
            x = torch.stack([load_template_tensor(split, inst, t) for t in ts]).to(DEVICE)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                z = tcia.encode_head(x, 'tmpl').float()
            for j, t in enumerate(ts):
                tz[t] = z[j]
                egap[t] = cache[str(t)]['gap'].to(DEVICE)
        out_f = open(out_path, 'a', encoding='utf-8')
        inst_new = 0
        for f in frames:
            trials = [r for r in protocol_frame_trials(split, inst, f, tmpls) if (r['t'], f) not in done_keys]
            if not trials:
                continue
            fr = load_frame(ROOT / split / inst, f)
            ir_bgr = letterboxed_bgr(split, inst, f)
            trial_data = []
            for s in range(0, len(trials), 12):
                sub = trials[s:s + 12]
                peak_list, crop_list = extract_peaks_and_crops(
                    locator, [cache[str(r['t'])] for r in sub], fr, ir_bgr)
                for r, peaks, crops in zip(sub, peak_list, crop_list):
                    labels = label_peaks(peaks, fr['gt_box'], fr['distractor_boxes'])
                    trial_data.append((r, peaks, crops, labels))
            uniq, seen = [], {}
            for r, peaks, crops, labels in trial_data:
                for (x, y, _v), crop in zip(peaks, crops):
                    key = (int(x), int(y))
                    if key not in seen:
                        seen[key] = len(uniq); uniq.append((key, crop))
            gaps, zs = [], []
            for s in range(0, len(uniq), 128):
                x = torch.stack([norm224(c) for _key, c in uniq[s:s + 128]]).to(DEVICE)
                gaps.append(tcia.encode_raw(x))
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    zs.append(tcia.encode_head(x, 'cand').float())
            gaps = torch.cat(gaps) if gaps else torch.empty(0, 2048, device=DEVICE)
            zs = torch.cat(zs) if zs else torch.empty(0, 256, device=DEVICE)
            cg = F.normalize(gaps, dim=1)
            ce = F.normalize(expert(gaps), dim=1)
            for r, peaks, crops, labels in trial_data:
                t = r['t']
                idx = torch.tensor([seen[(int(x), int(y))] for x, y, _v in peaks], device=DEVICE)
                cos = cg[idx] @ F.normalize(egap[t], dim=0)
                et = F.normalize(expert(egap[t].unsqueeze(0)), dim=1)[0]
                cos_e = ce[idx] @ et
                pn = torch.tensor(frame_minmax([float(p[2]) for p in peaks]), device=DEVICE)
                vpp = (0.3 * pn + 0.7 * cos_e).cpu().numpy()
                ts = (zs[idx] @ tz[t]).cpu().numpy()
                fus = fuse(vpp, ts, alpha)
                pv, pf = int(np.argmax(vpp)), int(np.argmax(fus))
                gt, obj = [float(v) for v in fr['gt_center']], [float(v) for v in fr['obj_size']]
                def hit(pi):
                    p = peaks[pi]
                    return bool(np.hypot(p[0] - gt[0], p[1] - gt[1]) <= 0.5 * max(obj))
                rec = {'inst': inst, 't': t, 'f': f, 'diff': fr['difficulty'],
                       'gt': [round(v, 2) for v in gt], 'obj': [round(v, 2) for v in obj],
                       'peaks': [[int(x), int(y), round(float(v), 7)] for x, y, v in peaks],
                       'labels': labels,
                       'vpp': [round(float(v), 7) for v in vpp],
                       'tcia': [round(float(v), 7) for v in ts],
                       'fusion': [round(float(v), 7) for v in fus],
                       'pick_vpp': pv, 'pick_fusion': pf,
                       'hit_vpp': hit(pv), 'hit_fusion': hit(pf)}
                out_f.write(json.dumps(rec, ensure_ascii=False) + '\n'); inst_new += 1
            out_f.flush()
            done_keys.update((r['t'], f) for r, _p, _c, _l in trial_data)
            if len(done_keys) % 500 < 20:
                print(f'[{split} {inst}] f={f} trials={len(done_keys)}/{expected_inst} '
                      f'elapsed={time.time()-t0:.0f}s', flush=True)
            del trial_data, uniq, gaps, zs, cg, ce
        out_f.close()
        assert len(done_keys) == expected_inst, (split, inst, len(done_keys), expected_inst)
        total_done += len(done_keys)
        print(f'[{split} {inst}] DONE trials={len(done_keys)} new={inst_new} '
              f'sha={sha256(out_path)} elapsed={time.time()-t0:.0f}s', flush=True)
        del cache, tz, egap
    assert total_done == expected_total, (total_done, expected_total)
    write_jsonl(complete_marker, [{'type': f'{split}_complete', 'n_trials': total_done,
                                  'alpha': alpha, 'elapsed_s': time.time() - t0,
                                  'epoch': state['epoch'], 'pipeline': 'B12 extract_peaks_and_crops + frame unique'}])
    summarize_split(split, alpha)


def summarize_split(split, alpha):
    rows = []
    for p in sorted(OUT.glob(score_glob(split))):
        rows.extend(json.loads(l) for l in open(p, encoding='utf-8'))
    expected = 197649 if split == 'val' else 307527
    assert len(rows) == expected and len({(r['inst'], r['t'], r['f']) for r in rows}) == expected
    def agg(key):
        def stat(sub):
            if not sub:
                return {'pair_macro': None, 'n_pairs': 0, 'n_trials': 0}
            groups = {}
            for r in sub:
                groups.setdefault((r['inst'], r['t']), []).append(int(r[key]))
            return {'pair_macro': float(np.mean([np.mean(v) for v in groups.values()])),
                    'n_pairs': len(groups), 'n_trials': len(sub)}
        unseen = [r for r in rows if r['inst'].startswith(('animal_', 'truck_'))]
        seen = [r for r in rows if not r['inst'].startswith(('animal_', 'truck_'))]
        return {'overall': stat(rows), 'easy': stat([r for r in rows if r['diff'] == 'easy']),
                'hard': stat([r for r in rows if r['diff'] == 'hard']),
                'seen': stat(seen), 'unseen': stat(unseen)}
    vpp, fus = agg('hit_vpp'), agg('hit_fusion')
    summary = {'type': f'{split}_summary', 'alpha': alpha, 'n_trials': len(rows),
               'vpp': vpp, 'fusion': fus,
               'hard_drop_pp': (vpp['hard']['pair_macro'] - fus['hard']['pair_macro']) * 100,
               'unseen_drop_pp': (vpp['unseen']['pair_macro'] - fus['unseen']['pair_macro']) * 100}
    if split == 'test':
        summary['stage_a_pass'] = bool(fus['overall']['pair_macro'] >= 0.4780
                                       and summary['hard_drop_pp'] <= 1 and summary['unseen_drop_pp'] <= 1)
    write_jsonl(artifact(f'{split}_summary'), [summary])
    files = sorted(OUT.glob(score_glob(split)))
    man = [{'type': f'{split}_manifest', 'alpha': alpha, 'n_trials': len(rows),
            'files': [{'file': p.name, 'sha256': sha256(p)} for p in files],
            'summary_sha256': sha256(artifact(f'{split}_summary'))}]
    write_jsonl(artifact(f'{split}_manifest'), man)
    print(f'{split.upper()}_SUMMARY', json.dumps(summary, ensure_ascii=False), flush=True)


def phase_full_val():
    a = json.loads(open(artifact('alpha'), encoding='utf-8').readlines()[-1])['alpha']
    full_pass('val', a)


def phase_full_test():
    a = json.loads(open(artifact('alpha'), encoding='utf-8').readlines()[-1])['alpha']
    assert (artifact('val_summary')).exists(), '先完成全量val'
    full_pass('test', a)


def phase_manifest():
    files = [CODE / 'models/tcia_v2.py', CODE / 'tools/tcia_v2_data.py',
             CODE / 'tools/tcia_v2_train.py', CODE / 'tools/tcia_v2_eval.py',
             CODE / 'configs/final/tcia_v2_s2024.yaml', TMPL_CACHE]
    files += sorted(p for p in OUT.glob(artifact_glob()) if p.name != 'tcia_v2_manifest.jsonl')
    ckdir = ckpt_path().parent
    files += sorted(ckdir.glob('*.pt')) if ckdir.exists() else []
    rows = []
    for p in files:
        if p.is_file():
            rows.append({'type': 'file', 'path': str(p.relative_to(CODE.parent)),
                         'bytes': p.stat().st_size, 'sha256': sha256(p)})
    rows.append({'type': 'env', 'torch': torch.__version__, 'n_cache_tmpl_sha256': sha256(TMPL_CACHE),
                 'test_completed': test_completed()})
    write_jsonl(artifact('manifest'), rows)
    print('MANIFEST', len(rows), flush=True)


def phase_by_inst():
    for split in ('val', 'test'):
        paths = sorted(OUT.glob(score_glob(split)))
        if not paths:
            continue
        rows = []
        for p in paths:
            rr = [json.loads(l) for l in open(p, encoding='utf-8')]
            inst = rr[0]['inst']
            def pm(key):
                groups = {}
                for r in rr:
                    groups.setdefault(r['t'], []).append(int(r[key]))
                return float(np.mean([np.mean(v) for v in groups.values()]))
            rows.append({'inst': inst, 'n_trials': len(rr), 'vpp': pm('hit_vpp'),
                         'fusion': pm('hit_fusion'), 'delta_pp': (pm('hit_fusion') - pm('hit_vpp')) * 100,
                         'file_sha256': sha256(p)})
        write_jsonl(artifact(f'{split}_by_inst'), rows)
        print(split, 'by_inst', len(rows), flush=True)


def self_test():
    assert SEED in (42, 123, 2024)
    name = artifact('val_summary').name
    expected = 'tcia_v2_val_summary.jsonl' if SEED == 42 else f'tcia_v2_s{SEED}_val_summary.jsonl'
    assert name == expected, (name, expected)
    complete_name = artifact('test_complete').name
    expected_complete = 'tcia_v2_test_complete.jsonl' if SEED == 42 else f'tcia_v2_s{SEED}_test_complete.jsonl'
    assert complete_name == expected_complete, (complete_name, expected_complete)
    expected_ckpt = 'best.pt' if SEED == 42 else f'tcia_v2_s{SEED}/best.pt'
    assert str(ckpt_path()).endswith(expected_ckpt), ckpt_path()
    if SEED != 42:
        assert not test_completed(), '非42 seed 不应复用 seed42 的 test 完成标记'
    print('EVAL_SEED_SELF_TEST_PASS', SEED, name, ckpt_path())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', required=True, choices=['g0', 'm0_alpha', 'full_val', 'full_test', 'by_inst', 'manifest', 'self_test'])
    ap.add_argument('--seed', type=int, default=None, choices=[42, 123, 2024])
    a = ap.parse_args()
    if a.seed is not None:
        global SEED
        SEED = a.seed
        os.environ['TCIA_SEED'] = str(SEED)
    {'g0': phase_g0, 'm0_alpha': phase_m0_alpha, 'full_val': phase_full_val,
     'full_test': phase_full_test, 'by_inst': phase_by_inst, 'manifest': phase_manifest, 'self_test': self_test}[a.phase]()


if __name__ == '__main__':
    main()
