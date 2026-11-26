# TCIA 训练：15 epoch、每 2 epoch 完整 val 选模（early stop 3）；训练/评测同一 top-20 峰 crop。
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
from models.tcia import TCIAv2
from tools.dataset_io import letterboxed_bgr, load_template_tensor
from tools.image_ops import norm224
from tools.locator import crops_from_peaks
from tools.paths import OUT, weights_dir, write_jsonl

DEVICE = 'cuda'
SEED = int(os.environ.get('TCIA_SEED', '0'))
MAX_EPOCHS = 15
EARLY_STOP_PATIENCE = 3
CKPT_DIR = Path(os.environ.get('TCIA_CKPT_DIR', str(weights_dir() / 'tcia')))
DATA_PATH = Path(os.environ.get('TCIA_TRAIN_TRIALS', str(OUT / 'tcia_train_trials.jsonl')))
TAG = os.environ.get('TCIA_TAG', '')


def artifact(name):
    tag = TAG or f's{SEED}'
    return OUT / f'tcia_{tag}_{name}.jsonl'


torch.manual_seed(SEED)


def load_records():
    if not DATA_PATH.exists():
        raise SystemExit(f'TCIA training trials not found: {DATA_PATH}\n'
                         'Run scripts/03_build_candidates.sh first.')
    return [json.loads(l) for l in open(DATA_PATH, encoding='utf-8')]


def collate_v2(rows):
    real, wrong, cx, labs = [], [], [], []
    for r in rows:
        real.append(load_template_tensor('train', r['inst'], r['t']))
        wrong.append(load_template_tensor(r['donor']['split'], r['donor']['inst'], r['donor']['t']))
        ir_bgr = letterboxed_bgr('train', r['inst'], r['f'])
        # 训练时把首个原始正峰确定性移到index0，其余顺序不变；labels/crops同步重排
        first_pos = r['labels'].index(2)
        order = [first_pos] + [i for i in range(20) if i != first_pos]
        peaks = [r['peaks'][i] for i in order]
        labels = [r['labels'][i] for i in order]
        crops = crops_from_peaks(ir_bgr, peaks)
        cx.append(torch.stack([norm224(c) for c in crops]))
        labs.append(torch.tensor(labels, dtype=torch.long))
    return {'real': torch.stack(real), 'wrong': torch.stack(wrong), 'cand': torch.cat(cx),
            'labels': torch.stack(labs),
            'real_keys': [f"{r['inst']}|{r['t']}" for r in rows],
            'wrong_keys': [f"{r['donor']['inst']}|{r['donor']['t']}" for r in rows]}


def episodic_batches(records, epoch):
    """同(inst,t)的两帧优先配对；2实例×2 trials，模板可去重，所有trial每轮一次。"""
    rng = np.random.RandomState(SEED + epoch)
    by_inst = {}
    for i, r in enumerate(records):
        by_inst.setdefault(r['inst'], []).append(i)
    chunks = []
    for inst in sorted(by_inst):
        by_t = {}
        for i in by_inst[inst]:
            by_t.setdefault(records[i]['t'], []).append(i)
        ids = []
        for t in sorted(by_t):
            tt = np.array(by_t[t]); rng.shuffle(tt); ids.extend(tt.tolist())
        for s in range(0, len(ids), 2):
            chunks.append((inst, ids[s:s + 2]))
    rng.shuffle(chunks)
    batches = []
    for s in range(0, len(chunks), 2):
        group = chunks[s:s + 2]
        batches.append([i for _inst, ids in group for i in ids])
    return batches


def unique_encode(model, x, keys, role):
    unique, inverse = [], []
    seen = {}
    for k in keys:
        if k not in seen:
            seen[k] = len(unique); unique.append(k)
        inverse.append(seen[k])
    idx = torch.tensor([seen[k] for k in keys], dtype=torch.long)
    # keys顺序与x一致；每个唯一键只前向一次
    first = [keys.index(k) for k in unique]
    z = model.encode_head(x[first], role)
    return z[idx]


def make_opt(model):
    lora = [p for n, p in model.named_parameters() if p.requires_grad and '.lora_' in n]
    head = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith('head.')]
    return torch.optim.AdamW([{'params': lora, 'lr': 1e-4}, {'params': head, 'lr': 1e-3}],
                             weight_decay=0.0, fused=True)


def run_batch(model, opt, batch):
    real = batch['real'].to(DEVICE, non_blocking=True)
    wrong = batch['wrong'].to(DEVICE, non_blocking=True)
    cand = batch['cand'].to(DEVICE, non_blocking=True)
    labels = batch['labels'].to(DEVICE)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        zr = unique_encode(model, real, batch['real_keys'], 'tmpl')
        zw = unique_encode(model, wrong, batch['wrong_keys'], 'tmpl')
        zc = model.encode_head(cand, 'cand')
    zr, zw, zc = zr.float(), zw.float(), zc.float()
    B, K = labels.shape
    zc = zc.view(B, K, -1)
    sc = (zc * zr.unsqueeze(1)).sum(-1)
    sw = (zc * zw.unsqueeze(1)).sum(-1)
    pos = labels == 2
    num = torch.logsumexp(sc.masked_fill(~pos, -float('inf')) / 0.07, dim=1)
    den = torch.logsumexp(sc / 0.07, dim=1)
    info = (den - num).mean()
    assert bool((labels[:, 0] == 2).all()), 'index0必须为GT正峰'
    real_gt, wrong_gt = sc[:, 0], sw[:, 0]
    wm = F.relu(0.2 - real_gt + wrong_gt).mean()
    loss = info + 0.5 * wm
    opt.zero_grad(); loss.backward(); opt.step()
    return {'loss': float(loss.detach()), 'info': float(info.detach()),
            'wrong': float(wm.detach())}


def loader(records, batches, workers=4):
    return DataLoader(records, batch_sampler=batches, collate_fn=collate_v2,
                      num_workers=workers, pin_memory=True, persistent_workers=False)


def assert_env():
    assert torch.cuda.is_available(), 'CUDA GPU required'
    free = int(os.popen("free -m | awk 'NR==2{print $7}'").read().strip())
    assert free >= 4000, free


def self_test():
    rows = [{'inst': 'a', 't': 0}, {'inst': 'a', 't': 1}, {'inst': 'b', 't': 0}, {'inst': 'b', 't': 1}]
    assert episodic_batches(rows, 0) == episodic_batches(rows, 0)
    print('TRAIN_SEED_SELF_TEST_PASS', SEED, artifact('train_log').name, CKPT_DIR.name)


def reconstruct_state(logs, val_rows):
    """恢复 best_score/stale/pending_val：依据 val_log 重建选模状态，避免断点续训重置。
    规则：best=(overall,hard,seen) 最大者；stale=最后一次提升后已评 epoch 数；
    pending=train 已到奇数 epoch 但 val 缺失（含末 epoch）时该 epoch。"""
    val_epochs = {r['epoch'] for r in val_rows}
    best_score, last_improve = None, None
    for r in sorted(val_rows, key=lambda r: r['epoch']):
        score = (r['hit_tcia_overall'], r['hit_tcia_hard'], r['hit_tcia_seen'])
        if best_score is None or score > best_score:
            best_score, last_improve = score, r['epoch']
    stale = len([e for e in val_epochs if last_improve is not None and e > last_improve]) if last_improve is not None else 0
    pending = None
    if logs:
        last_ep = logs[-1]['epoch']
        if (last_ep % 2 == 1 or last_ep == MAX_EPOCHS - 1) and last_ep not in val_epochs:
            pending = last_ep
    return best_score, stale, pending


def _save_ckpt(path, epoch, model, opt, val_row=None):
    payload = {'epoch': epoch, 'state': model.adapter_state_dict(),
               'opt': opt.state_dict() if opt is not None else None,
               'model_version': model.MODEL_VERSION}
    if val_row is not None:
        payload['val'] = val_row
    torch.save(payload, path)


def phase_train():
    assert_env(); records = load_records(); model = TCIAv2().to(DEVICE).train(); opt = make_opt(model)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    log_path = artifact('train_log')
    logs = [json.loads(l) for l in open(log_path, encoding='utf-8')] if log_path.exists() else []
    start = 0
    val_log_path = artifact('val_log')
    val_rows = [json.loads(l) for l in open(val_log_path, encoding='utf-8')] if val_log_path.exists() else []
    pending_val = None
    if logs:
        start = logs[-1]['epoch'] + 1
        state = torch.load(CKPT_DIR / f'epoch_{logs[-1]["epoch"]:02d}.pt', map_location=DEVICE)
        model.load_adapter_state_dict(state['state']); opt.load_state_dict(state['opt'])
        best_score, stale, pending_val = reconstruct_state(logs, val_rows)
        print('[resume]', start, 'best=', best_score, 'stale=', stale, 'pending_val=', pending_val, flush=True)
    t0 = time.time()
    if not logs:
        best_score, stale = None, 0
    ep_iter = range(start, MAX_EPOCHS)
    if pending_val is not None:
        # 补评被中断的 epoch：载入对应权重评测后并入主循环计数
        torch.cuda.empty_cache()
        st = torch.load(CKPT_DIR / f'epoch_{pending_val:02d}.pt', map_location=DEVICE)
        model_p = TCIAv2().to(DEVICE).eval()
        model_p.load_adapter_state_dict(st['state'])
        from tools.tcia_eval import run_split as _rs
        _vs = _rs(model_p, 'val', write_scores=False)[0]
        del model_p
        _vr = {'type': 'val_epoch_full', 'epoch': pending_val,
               'hit_tcia_overall': _vs['hit_tcia_overall'], 'hit_tcia_hard': _vs['hit_tcia_hard'],
               'hit_tcia_seen': _vs['hit_tcia_seen'], 'hit_tcia_unseen': _vs['hit_tcia_unseen'],
               'hit_locator_overall': _vs['hit_locator_overall'], 'n_trials': _vs['n_trials'],
               'elapsed_s': _vs['elapsed_s']}
        write_jsonl(val_log_path, [_vr], mode='a')
        _sc = (_vr['hit_tcia_overall'], _vr['hit_tcia_hard'], _vr['hit_tcia_seen'])
        if best_score is None or _sc > best_score:
            best_score, stale = _sc, 0
            torch.save({'epoch': pending_val, 'state': st['state'], 'opt': st.get('opt'),
                        'val': _vr, 'model_version': model.MODEL_VERSION},
                       CKPT_DIR / 'best_fullval.pt')
            print('BEST_FULLVAL_UPDATED(pending)', _vr, flush=True)
        else:
            stale += 1
            print('VAL_NO_IMPROVE(pending)', _vr, f'stale={stale}', flush=True)
        del st
    for ep in ep_iter:
        batches = episodic_batches(records, ep)
        stats, done = [], 0
        for bi, batch in enumerate(loader(records, batches)):
            stats.append(run_batch(model, opt, batch)); done += len(batch['labels'])
            if (bi + 1) % 100 == 0:
                print(f'[ep{ep}] {done}/{len(records)} loss={np.mean([x["loss"] for x in stats[-100:]]):.5f} '
                      f'elapsed={time.time()-t0:.0f}s vram={torch.cuda.max_memory_allocated()/1e9:.2f}GB', flush=True)
        assert done == len(records), (done, len(records))
        row = {'type': 'epoch', 'epoch': ep, 'n_trials': done,
               **{k: float(np.mean([x[k] for x in stats])) for k in ('loss', 'info', 'wrong')},
               'elapsed_s': time.time() - t0, 'vram_gb': torch.cuda.max_memory_allocated()/1e9}
        logs.append(row)
        _save_ckpt(CKPT_DIR / f'epoch_{ep:02d}.pt', ep, model, opt)
        write_jsonl(log_path, logs)
        print('EPOCH_DONE', row, flush=True)
        if ep % 2 == 1 or ep == MAX_EPOCHS - 1:
            from tools.tcia_eval import run_split as fullval_run
            torch.cuda.empty_cache()
            model.eval()
            val_summary = fullval_run(model, 'val', write_scores=False)[0]
            model.train()
            val_row = {'type': 'val_epoch_full', 'epoch': ep,
                       'hit_tcia_overall': val_summary['hit_tcia_overall'],
                       'hit_tcia_hard': val_summary['hit_tcia_hard'],
                       'hit_tcia_seen': val_summary['hit_tcia_seen'],
                       'hit_tcia_unseen': val_summary['hit_tcia_unseen'],
                       'hit_locator_overall': val_summary['hit_locator_overall'],
                       'n_trials': val_summary['n_trials'],
                       'elapsed_s': val_summary['elapsed_s']}
            write_jsonl(artifact('val_log'), [val_row], mode='a')
            score = (val_row['hit_tcia_overall'], val_row['hit_tcia_hard'], val_row['hit_tcia_seen'])
            improved = best_score is None or score > best_score
            if improved:
                best_score, stale = score, 0
                _save_ckpt(CKPT_DIR / 'best_fullval.pt', ep, model, opt, val_row=val_row)
                print('BEST_FULLVAL_UPDATED', val_row, flush=True)
            else:
                stale += 1
                print('VAL_NO_IMPROVE', val_row, f'stale={stale}/{EARLY_STOP_PATIENCE}', flush=True)
            if stale >= EARLY_STOP_PATIENCE:
                print('EARLY_STOP', ep, flush=True)
                break
    assert (CKPT_DIR / 'best_fullval.pt').exists(), '缺完整val选出的best_fullval.pt'
    print('TCIA_TRAIN DONE', flush=True)


def phase_unit():
    assert_env(); records = load_records(); model = TCIAv2().to(DEVICE).train(); opt = make_opt(model)
    batch = next(iter(loader(records, episodic_batches(records, 0)[:1], workers=0)))
    assert bool((batch['labels'][:, 0] == 2).all())
    stat = run_batch(model, opt, batch)
    assert np.isfinite(stat['loss'])
    print('UNIT_PASS index0=all_positive, loss=info+0.5*wrong', stat, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', required=True, choices=['unit', 'train', 'self_test'])
    ap.add_argument('--seed', type=int, default=None)
    a = ap.parse_args()
    if a.seed is not None:
        global SEED, CKPT_DIR
        SEED = a.seed
        if not os.environ.get('TCIA_CKPT_DIR'):
            CKPT_DIR = weights_dir() / 'tcia'
        os.environ['TCIA_SEED'] = str(SEED)
        torch.manual_seed(SEED)
    {'unit': phase_unit, 'train': phase_train, 'self_test': self_test}[a.phase]()


if __name__ == '__main__':
    main()
