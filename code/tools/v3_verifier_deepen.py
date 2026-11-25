# 实验 V3：判别验证器深挖——多尺度裁窗(M) × 模板 per-slice max(T) × 标定融合(C)
#
# 任务书 R-V3（零训练纯推理；判读分支预注册 §4，executor 无权自选结论）。
# 因子（10 臂全部由同一套缓存张量聚合，禁止重复推理）：
#   M（多尺度裁窗）：每峰以峰为中心裁 s×s（s∈{128,224,320}，越界补零，resize 224×224
#      进编码器；s=224 逐位复用 V 的 crop_patch）；尺度聚合 = 三尺度 cos 取 max；对照=仅 224。
#      禁止用 GT 尺寸选尺度（泄漏）。
#   T（模板聚合）：模板侧保留每张切片嵌入，cos 逐切片算后取 max；对照=切片均值嵌入 cos（V 口径）。
#   C（标定融合）：logistic p=σ(w1·peak_norm + w2·cos + b)，train split 拟合（每峰一样本，
#      特征=[peak_norm, cos]，标签=该峰是否命中 GT；torch Adam/BCE/2000 步/seed 0/L2=1e-4/
#      不做类别重加权；lr 未指定 → Adam 默认 1e-3，报告注明），冻结后 val 上 argmax p；
#      对照=线性 α=0.8（V 整改口径）。peak_norm=帧内 5 峰 sigmoid 值 min-max（span<1e-9 全置 0），
#      行为逐位复用 tools/v_verifier_rerank._norm_fusion_pred。
# 存储：候选嵌入 fp32 (5,3,256)——理由：A1 锚与 V 的 cos 逐位可比（fp16 舍入 ~5e-4 会威胁
#      ±0.002 锚容差）且体积小（val ~26MB / train ~88MB）；模板切片 fp16 + 每实例 fp32 均值
#      （均值由 fp32 嵌入算出，与 V 的 v_template_emb.pt 同口径，保障 A1 逐位）。
# 纪律：解释器 viir；CUDA_VISIBLE_DEVICES=0（默认 FASTEST_FIRST 排序 → 物理槽位1的 4090，
#      nvidia-smi index 1——已实测确认，见报告）；fp32+eval+no_grad；torch.manual_seed(0)；
#      test 封存；零覆写（本文件与 v3_* 产物全部新建）。
#
# 用法:
#   python tools/v3_verifier_deepen.py --g0a                 # G0a 纯数据锚（必须先过）
#   python tools/v3_verifier_deepen.py --tmpl                # 模板 per-slice 缓存（val+train 93 实例）
#   python tools/v3_verifier_deepen.py --smoke               # 20 帧冒烟
#   python tools/v3_verifier_deepen.py --stage1 --split val   # val 1736 帧全量缓存
#   python tools/v3_verifier_deepen.py --stage1 --split train # train 5738 帧全量缓存
#   python tools/v3_verifier_deepen.py --stage2               # 10 臂 + logistic + 判读 + 案例
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
from data.cstloc_dataset import CSTLocDataset, _letterbox            # noqa: E402
from models.localizer import Localizer                                # noqa: E402
from models.encoders.dinov2_encoder import DINOv2TemplateEncoder      # noqa: E402
from models.identity_expert import IdentityExpert                     # noqa: E402
from tools.v_verifier_rerank import (ROOT, EXPERT_CKPT, MODEL_CKPT,   # noqa: E402
    IMG_SIZE, PATCH, K, MIN_DIST, PCK_THRESH, OUT_DIR, _Idx, nms_peaks,
    crop_patch, norm224, _norm_fusion_pred)

SCALES = [128, 224, 320]
S224_IDX = SCALES.index(224)
ALPHA = 0.8
ANCHOR_A0 = 0.4412
ANCHOR_A1 = (0.4643, 0.6411, 0.3712)
TMPL_BATCH = 64
CROP_BATCH = 64          # 裁窗前向 batch≤64（任务书红线）
HEAT_BATCH = 8           # 基准前向 batch≤8（任务书红线）
UNSEEN = ('animal_', 'truck_')
DEVICE = 'cuda'


def pck_hit(pred, gt, obj):
    return float(np.hypot(pred[0] - gt[0], pred[1] - gt[1])) <= PCK_THRESH * max(obj[0], obj[1])


def crop_patch_s(ir_bgr, cx, cy, s):
    """s×s 裁窗（峰中心，越界补零）→ resize 224×224。s=224 逐位复用 V 的 crop_patch（不 resize）。"""
    if s == PATCH:
        return crop_patch(ir_bgr, cx, cy)
    H, W = ir_bgr.shape[:2]
    hw = s // 2
    x0, y0 = int(cx) - hw, int(cy) - hw
    x1, y1 = x0 + s, y0 + s
    x0c, y0c = max(0, x0), max(0, y0)
    x1c, y1c = min(W, x1), min(H, y1)
    win = np.zeros((s, s, 3), dtype=np.uint8)
    if x1c > x0c and y1c > y0c:
        win[y0c - y0:y1c - y0, x0c - x0:x1c - x0] = ir_bgr[y0c:y1c, x0c:x1c]
    rgb = cv2.cvtColor(win, cv2.COLOR_BGR2RGB)
    return cv2.resize(rgb, (PATCH, PATCH), interpolation=cv2.INTER_LINEAR)


def load_model_and_ds(split):
    """j2_expert 定位模型 + 数据集（mode='test' 确定性口径，与 V 完全同构）。"""
    torch.manual_seed(0)
    cfg = yaml.safe_load(open(CODE / 'configs' / 'v2_s1_seed42.yaml', encoding='utf-8'))
    ds = _Idx(CSTLocDataset(root=str(ROOT), split=split, mode='test',
                            img_size=tuple(cfg['img_size']), patch_size=cfg['patch_size'],
                            sigma_factor=cfg['sigma_factor'], n_templates=cfg['eval_n_templates'],
                            min_gap_frames=cfg['min_gap_frames'], instances=None))
    model = Localizer(encoder='dinov2', modulator='correlation', pretrained=True,
                      pretrained_path=cfg.get('pretrained_path'), ir_encoder='infmae',
                      use_id=True, identity_expert=str(EXPERT_CKPT), mask_null_p=0.0).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_CKPT, map_location='cpu'), strict=True)
    model.eval()
    enc = DINOv2TemplateEncoder(pretrained=True).cuda().eval()
    expert = IdentityExpert(str(EXPERT_CKPT)).cuda().eval()
    return cfg, ds, model, enc, expert


# ---------------- G0a：纯数据锚 ----------------
def g0a():
    doc = json.load(open(OUT_DIR / 'v_rerank_val.json', encoding='utf-8'))
    frames = doc['frames']
    agg = {'overall': [0, 0], 'easy': [0, 0], 'hard': [0, 0]}
    for fr in frames:
        pred, _ = _norm_fusion_pred(fr, ALPHA)     # 逐位复用 V 整改口径
        hit = pck_hit(pred, fr['gt'], fr['obj'])
        agg['overall'][0] += int(hit); agg['overall'][1] += 1
        d = fr['difficulty']
        if d in agg:
            agg[d][0] += int(hit); agg[d][1] += 1
    o = agg['overall'][0] / agg['overall'][1]
    e = agg['easy'][0] / agg['easy'][1]
    h = agg['hard'][0] / agg['hard'][1]
    print(f'G0a: α={ALPHA} 归一化融合（纯 v_rerank_val.json 数据重算）')
    print(f'  overall={o:.4f} easy={e:.4f} hard={h:.4f}')
    print(f'  锚: {ANCHOR_A1[0]:.4f}/{ANCHOR_A1[1]:.4f}/{ANCHOR_A1[2]:.4f}（±0.0006）')
    ok = (abs(o - ANCHOR_A1[0]) <= 6e-4 and abs(e - ANCHOR_A1[1]) <= 6e-4
          and abs(h - ANCHOR_A1[2]) <= 6e-4)
    print(f'G0a: {"PASS" if ok else "FAIL"}')
    out = {'alpha': ALPHA, 'n_frames': len(frames),
           'overall': round(o, 4), 'easy': round(e, 4), 'hard': round(h, 4),
           'anchor': list(ANCHOR_A1), 'tol': 6e-4, 'pass': bool(ok),
           'fusion': 'score=(1-a)*peak_norm(frame-wise minmax, span<1e-9 -> 0) + a*cos_raw'}
    (OUT_DIR / 'v3_g0a.json').write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                         encoding='utf-8')
    if not ok:
        sys.exit(1)


# ---------------- 模板 per-slice 缓存（val 23 + train 70）----------------
def tmpl_cache():
    torch.manual_seed(0)
    enc = DINOv2TemplateEncoder(pretrained=True).cuda().eval()
    expert = IdentityExpert(str(EXPERT_CKPT)).cuda().eval()
    slices, means, counts, split_of = {}, {}, {}, {}
    t0 = time.time()
    for split in ('val', 'train'):
        insts = sorted(p.name for p in (ROOT / split).iterdir() if p.is_dir())
        for inst in insts:
            files = sorted((ROOT / split / inst / 'rgb' / 'template').glob('*.jpg'),
                           key=lambda q: int(q.stem))
            embs = []
            with torch.no_grad():
                for s in range(0, len(files), TMPL_BATCH):
                    # 与 V 的模板缓存逐位同口径：cv2.resize 224 INTER_LINEAR → norm224
                    xs = torch.stack([norm224(cv2.resize(
                        cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB),
                        (PATCH, PATCH), interpolation=cv2.INTER_LINEAR))
                        for p in files[s:s + TMPL_BATCH]]).to(DEVICE)
                    e = expert(enc.intermediate_gap(xs)).cpu()
                    embs.append(e)
            embs = torch.cat(embs)                       # (N,256) fp32
            slices[inst] = embs.to(torch.float16)
            means[inst] = embs.mean(0)                   # fp32 均值（V 口径）
            counts[inst] = int(embs.shape[0])
            split_of[inst] = split
        print(f'[{split}] {len(insts)} 实例模板切片完成（累计 {sum(counts.values())} 张, '
              f'{time.time() - t0:.0f}s）')
    out = {'meta': {'expert': str(EXPERT_CKPT), 'gap': 'intermediate_gap(taps=[15,23]) -> 2048',
                    'slices_dtype': 'fp16', 'mean_dtype': 'fp32(由 fp32 嵌入求均值，V 口径)',
                    'counts': counts, 'split_of': split_of,
                    'total_slices': sum(counts.values())},
           'slices': slices, 'mean_fp32': means}
    torch.save(out, OUT_DIR / 'v3_template_slices.pt')
    print(f'落盘: v3_template_slices.pt（{sum(counts.values())} 张切片 / {len(counts)} 实例, '
          f'{time.time() - t0:.0f}s）')


# ---------------- 逐帧管线（冒烟与 stage1 共用）----------------
def frame_records(split, ds, model, n_limit=None):
    """基准前向（batch≤8）→ 每帧记录：peaks/baseline_pred/gt/obj/difficulty/inst/frame_id。"""
    loader = DataLoader(ds, batch_size=HEAT_BATCH, shuffle=False, num_workers=4, pin_memory=True)
    recs = {}
    t0 = time.time()
    n_done = 0
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            ir = batch['ir'].to(DEVICE)
            templates = batch['templates'].to(DEVICE)
            logits = model(ir, templates)
            heat = torch.sigmoid(logits)[:, 0].float()
            for i in range(ir.size(0)):
                idx = int(batch['_idx'][i])
                hn = heat[i].cpu().numpy()
                ay, ax = np.unravel_index(int(np.argmax(hn)), hn.shape)
                peaks = nms_peaks(hn, K, MIN_DIST)
                recs[idx] = {'idx': idx, 'inst': batch['instance'][i],
                             'frame_id': int(ds.samples[idx][1].stem),
                             'difficulty': batch['difficulty'][i],
                             'gt': [float(batch['gt_center'][i][0]), float(batch['gt_center'][i][1])],
                             'obj': [float(batch['obj_size'][i][0]), float(batch['obj_size'][i][1])],
                             'baseline_pred': [float(ax), float(ay)],
                             'peaks': [[float(p[0]), float(p[1]), float(p[2])] for p in peaks]}
                if n_limit and len(recs) >= n_limit:
                    break
            n_done += int(ir.size(0))
            if bi % 20 == 0:
                rate = n_done / (time.time() - t0)
                eta = (len(ds) - n_done) / max(rate, 1e-9)
                print(f'  热图 {n_done}/{len(ds)} ({rate:.1f} 帧/s, ETA {eta/60:.1f}min)')
            if n_limit and len(recs) >= n_limit:
                break
    print(f'  热图完成 {len(recs)} 帧, {time.time() - t0:.0f}s')
    return [recs[i] for i in sorted(recs)]


def crop_embeddings(recs, ds, enc, expert, tag):
    """每帧 5 峰 × 3 尺度裁窗 → (n_peaks,3,256) fp32 嵌入。重读 raw IR，裁窗前向 batch≤64。"""
    todo = []            # (rec_idx, peak_idx, scale_idx, patch_rgb)
    for ri, rec in enumerate(recs):
        inst_dir, sj, _ = ds.samples[rec['idx']]
        ir_raw = cv2.cvtColor(cv2.imread(str(sj.with_suffix('.jpg'))), cv2.COLOR_BGR2RGB)
        ir_raw, _, _, _ = _letterbox(ir_raw, IMG_SIZE)
        ir_bgr = cv2.cvtColor(ir_raw, cv2.COLOR_RGB2BGR)
        for pi, (x, y, _) in enumerate(rec['peaks']):
            for si, s in enumerate(SCALES):
                todo.append((ri, pi, si, crop_patch_s(ir_bgr, x, y, s)))
    print(f'  [{tag}] 裁窗总数 {len(todo)}，过 GAP+专家（batch {CROP_BATCH}）...')
    embs = [np.zeros((len(r['peaks']), len(SCALES), 256), dtype=np.float32) for r in recs]
    t0 = time.time()
    n_done = 0
    with torch.no_grad():
        for s in range(0, len(todo), CROP_BATCH):
            chunk = todo[s:s + CROP_BATCH]
            xs = torch.stack([norm224(c[3]) for c in chunk]).to(DEVICE)
            e = expert(enc.intermediate_gap(xs)).cpu().numpy()
            for j, (ri, pi, si, _) in enumerate(chunk):
                embs[ri][pi, si] = e[j]
            n_done += len(chunk)
            if (s // CROP_BATCH) % 50 == 0:
                rate = n_done / (time.time() - t0)
                eta = (len(todo) - n_done) / max(rate, 1e-9)
                print(f'  [{tag}] 裁窗 {n_done}/{len(todo)} ({rate:.0f} 窗/s, ETA {eta/60:.1f}min)')
    print(f'  [{tag}] 裁窗嵌入完成, {time.time() - t0:.0f}s')
    for rec, emb in zip(recs, embs):
        rec['cand_emb'] = torch.from_numpy(emb)      # fp32 (n_peaks,3,256)
    return recs


def stage1(split):
    cfg, ds, model, enc, expert = load_model_and_ds(split)
    print(f'[{split}] 样本数: {len(ds)} | 模型 {MODEL_CKPT.name} | 专家 {EXPERT_CKPT.name}')
    recs = frame_records(split, ds, model)
    recs = crop_embeddings(recs, ds, enc, expert, split)
    out = {'meta': {'model': str(MODEL_CKPT), 'expert': str(EXPERT_CKPT), 'split': split,
                    'scales': SCALES, 'k': K, 'min_dist': MIN_DIST, 'patch': PATCH,
                    'emb_dtype': 'fp32', 'n_frames': len(recs),
                    'pck_def': '0.5*max(w,h)'},
           'frames': recs}
    torch.save(out, OUT_DIR / f'v3_{split}_cache.pt')
    n_pk = sum(len(r['peaks']) for r in recs)
    summary = {'split': split, 'n_frames': len(recs), 'n_peaks': n_pk,
               'scales': SCALES, 'emb': 'fp32 (5,3,256)/帧',
               'cache_file': f'v3_{split}_cache.pt'}
    (OUT_DIR / f'v3_{split}_summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'落盘: v3_{split}_cache.pt + v3_{split}_summary.json（{len(recs)} 帧 / {n_pk} 峰）')


# ---------------- 20 帧冒烟 ----------------
def smoke(n=20):
    _, ds, model, enc, expert = load_model_and_ds('val')
    tmpl = torch.load(OUT_DIR / 'v3_template_slices.pt', map_location='cpu')
    v_tmpl = torch.load(OUT_DIR / 'v_template_emb.pt', map_location='cpu')   # V 的均值嵌入
    recs = frame_records('val-smoke', ds, model, n_limit=n)
    recs = crop_embeddings(recs, ds, enc, expert, 'smoke')
    # 均值嵌入 vs V 缓存一致性（fp32 均值应逐位一致）
    diffs = [float((tmpl['mean_fp32'][k] - v_tmpl[k]).abs().max()) for k in v_tmpl]
    print(f'均值嵌入 vs v_template_emb.pt 逐实例 max|Δ|（max over 23 实例）= {max(diffs):.2e}')
    n_hit = 0
    for rec in recs:
        sl = tmpl['slices'][rec['inst']].float()             # (N,256)
        mean = tmpl['mean_fp32'][rec['inst']]                # (256,)
        ce = rec['cand_emb']                                  # (n,3,256)
        cos_mean = F.cosine_similarity(ce, mean.unsqueeze(0).unsqueeze(0), dim=-1)   # (n,3)
        cos_sl = torch.einsum('psc,nc->psn', F.normalize(ce, dim=-1),
                              F.normalize(sl, dim=-1)).max(-1).values                     # (n,3)
        c224_mean = float(cos_mean[:, S224_IDX].max())
        cmax_mean = float(cos_mean.max())
        c224_slmax = float(cos_sl[:, S224_IDX].max())
        gt_hit0 = pck_hit(rec['peaks'][0][:2], rec['gt'], rec['obj'])
        oracle = any(pck_hit(p[:2], rec['gt'], rec['obj']) for p in rec['peaks'])
        n_hit += int(gt_hit0)
        pk = ' '.join(f'({int(p[0])},{int(p[1])},{p[2]:.3f})' for p in rec['peaks'])
        print(f"{rec['inst']} f{rec['frame_id']} {rec['difficulty']}: peaks {pk}")
        print(f"   cos[224/mean]={c224_mean:+.4f} cos[max3/mean]={cmax_mean:+.4f} "
              f"cos[224/slicemax]={c224_slmax:+.4f} | top1_hit={gt_hit0} oracle5={oracle}")
    print(f'冒烟 {len(recs)} 帧: top1 命中 {n_hit}（人工核验峰坐标/cos 合理性）')


# ---------------- 阶段 2：10 臂 + logistic + 判读 ----------------
def frame_minmax(vals):
    lo, hi = min(vals), max(vals)
    span = hi - lo
    if span > 1e-9:
        return [(v - lo) / span for v in vals]
    return [0.0] * len(vals)


def fit_logistic(feat, label, steps=2000, seed=0, l2=1e-4, lr=1e-3):
    """logistic p=σ(w1·pn + w2·cos + b)；torch Adam+BCE，2000 步，seed 0，L2=1e-4（weight_decay），
    不做类别重加权；lr 未在任务书指定 → Adam 默认 1e-3（报告注明）。"""
    torch.manual_seed(seed)
    x = torch.tensor(feat, dtype=torch.float32)
    y = torch.tensor(label, dtype=torch.float32)
    w = torch.zeros(2, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=lr, weight_decay=l2)
    for _ in range(steps):
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(x @ w + b, y)
        loss.backward()
        opt.step()
    return [float(w[0]), float(w[1])], float(b[0])


def arm_pck(recs, cos_key, score_fn):
    """score_fn(peak_vals, cos_list) -> 选择峰 idx；返回 overall/easy/hard 与逐帧选择。"""
    agg = {'overall': [0, 0], 'easy': [0, 0], 'hard': [0, 0]}
    picks = []
    for rec in recs:
        vals = [p[2] for p in rec['peaks']]
        cos_list = rec['feat'][cos_key] if cos_key is not None else None
        best = score_fn(vals, cos_list)
        pred = rec['peaks'][best][:2]
        hit = pck_hit(pred, rec['gt'], rec['obj'])
        agg['overall'][0] += int(hit); agg['overall'][1] += 1
        d = rec['difficulty']
        if d in agg:
            agg[d][0] += int(hit); agg[d][1] += 1
        picks.append(best)
    return agg, picks


def linear_fn(alpha=ALPHA):
    def fn(vals, cos_list):
        pn = frame_minmax(vals)
        best, bs = 0, -1e18
        for i in range(len(vals)):
            s = (1 - alpha) * pn[i] + alpha * cos_list[i]
            if s > bs:
                bs, best = s, i
        return best
    return fn


def peak_fn(vals, cos_list):
    best, bv = 0, -1e18
    for i, v in enumerate(vals):
        if v > bv:
            bv, best = v, i
    return best


def make_logistic_fn(w1, w2, b):
    def fn(vals, cos_list):
        pn = frame_minmax(vals)
        best, bs = 0, -1e18
        for i in range(len(vals)):
            s = w1 * pn[i] + w2 * cos_list[i] + b
            if s > bs:
                bs, best = s, i
        return best
    return fn


def stage2():
    tmpl = torch.load(OUT_DIR / 'v3_template_slices.pt', map_location='cpu')
    caches = {}
    for split in ('val', 'train'):
        doc = torch.load(OUT_DIR / f'v3_{split}_cache.pt', map_location='cpu')
        caches[split] = doc['frames']
        print(f"[{split}] 缓存 {len(doc['frames'])} 帧")
    # 逐帧特征：4 组 (M,T) cos（峰值嵌入 fp32 → fp32 cos）
    combos = [('224', 'mean'), ('224', 'max'), ('max', 'mean'), ('max', 'max')]
    cos_keys = {c: f"cos_{c[0]}_{c[1]}" for c in combos}
    for split in ('val', 'train'):
        for rec in caches[split]:
            sl = tmpl['slices'][rec['inst']].float()          # (N,256)
            mean = tmpl['mean_fp32'][rec['inst']]             # (256,)
            ce = rec['cand_emb']                               # (n,3,256) fp32
            cm = F.cosine_similarity(ce, mean.view(1, 1, -1), dim=-1)          # (n,3)
            cs = torch.einsum('psc,nc->psn', F.normalize(ce, dim=-1),
                              F.normalize(sl, dim=-1)).max(-1).values            # (n,3)
            rec['feat'] = {
                cos_keys[('224', 'mean')]: [float(v) for v in cm[:, S224_IDX]],
                cos_keys[('max', 'mean')]: [float(v) for v in cm.max(dim=1).values],
                cos_keys[('224', 'max')]: [float(v) for v in cs[:, S224_IDX]],
                cos_keys[('max', 'max')]: [float(v) for v in cs.max(dim=1).values],
            }
    # logistic：train 上 4 组权重（每峰一样本，标签=峰命中 GT）
    weights = {}
    for c in combos:
        k = cos_keys[c]
        feat, label = [], []
        for rec in caches['train']:
            pn = frame_minmax([p[2] for p in rec['peaks']])
            for i, p in enumerate(rec['peaks']):
                feat.append([pn[i], rec['feat'][k][i]])
                label.append(int(pck_hit(p[:2], rec['gt'], rec['obj'])))
        (w1, w2), b = fit_logistic(feat, label)
        weights[f'{c[0]}_{c[1]}'] = {'w1': w1, 'w2': w2, 'b': b, 'n_train_peaks': len(label),
                                     'train_label_rate': round(float(np.mean(label)), 4)}
        weights[f'{c[0]}_{c[1]}']['combo'] = list(c)
        print(f"logistic {c}: w1={w1:.4f} w2={w2:.4f} b={b:.4f} "
              f"(train 峰命中基率 {weights[f'{c[0]}_{c[1]}']['train_label_rate']})")
    # 10 臂评测
    arms = {}
    arms['A0_peak'] = {'score': 'argmax peak_val（纯峰锚）', 'fn': peak_fn, 'cos': None}
    for c in combos:
        k = cos_keys[c]
        tag = f'lin_M{c[0]}_T{c[1]}'
        arms[tag] = {'score': f'线性 α=0.8: 0.2·peak_norm + 0.8·cos({c[0]}裁窗,{c[1]}模板)',
                     'fn': linear_fn(), 'cos': k}
        tag = f'cal_M{c[0]}_T{c[1]}'
        w = weights[f'{c[0]}_{c[1]}']
        arms[tag] = {'score': f'logistic σ({w["w1"]:.4f}·pn + {w["w2"]:.4f}·cos + {w["b"]:.4f})',
                     'fn': make_logistic_fn(w['w1'], w['w2'], w['b']), 'cos': k}
    table = {}
    picks_of = {}
    for name, a in arms.items():
        agg, picks = arm_pck(caches['val'], a['cos'], a['fn'])
        table[name] = {'overall': round(agg['overall'][0] / agg['overall'][1], 4),
                       'easy': round(agg['easy'][0] / agg['easy'][1], 4),
                       'hard': round(agg['hard'][0] / agg['hard'][1], 4),
                       'n': agg['overall'][1], 'score': a['score']}
        picks_of[name] = picks
        print(f"{name:18s} overall={table[name]['overall']:.4f} easy={table[name]['easy']:.4f} "
              f"hard={table[name]['hard']:.4f} | {a['score']}")
    # logistic 选择器在 train 自身上的 PCK（标定跨实例泛化 gap）
    train_pck = {}
    for c in combos:
        w = weights[f'{c[0]}_{c[1]}']
        agg, _ = arm_pck(caches['train'], cos_keys[c], make_logistic_fn(w['w1'], w['w2'], w['b']))
        train_pck[f'cal_M{c[0]}_T{c[1]}'] = round(agg['overall'][0] / agg['overall'][1], 4)
        print(f"  {c} 标定选择器在 train 自身: PCK={train_pck[f'cal_M{c[0]}_T{c[1]}']} "
              f"（val 对照见总表）")
    # 最优臂（防挑数规则：overall 降序；±0.0005 并列取 hard 高者）
    order = sorted(table.items(), key=lambda kv: (-kv[1]['overall'], -kv[1]['hard']))
    top_overall = order[0][1]['overall']
    tied = [kv for kv in order if abs(kv[1]['overall'] - top_overall) <= 5e-4]
    best_name, best = max(tied, key=lambda kv: kv[1]['hard'])
    sel_log = [f'{n}: o={t["overall"]} h={t["hard"]}' for n, t in order[:5]]
    print('最优臂选择（overall 排序前5）: ' + ' | '.join(sel_log))
    print(f'最优臂 = {best_name} {best}（并列容差 ±0.0005 内取 hard 高者）')
    # G1 判读（预注册）
    if best['overall'] >= 0.4843 and best['hard'] >= 0.3812:
        g1 = '(i) 验证器升级主贡献（best≥0.4843 且 hard≥0.3812）'
    elif best['overall'] > 0.4643:
        g1 = '(ii) 温和改善（0.4643<best<0.4843），论文默认口径仍记 V（0.4643）'
    else:
        g1 = '(iii) 三因子均无效（best≤0.4643），归档，验证器以 V 口径进消融'
    n_pass = sum(1 for t in table.values() if t['overall'] >= 0.4643)
    print(f'G1 判定: {g1}')
    print(f'稳健性画像: 10 臂中 overall≥0.4643 的臂数 = {n_pass}')
    # 最优臂 per-class
    pc = {}
    for rec, pk in zip(caches['val'], picks_of[best_name]):
        cls = rec['inst'].split('_')[0] + '_'
        pc.setdefault(cls, [0, 0])
        pc[cls][0] += int(pck_hit(rec['peaks'][pk][:2], rec['gt'], rec['obj']))
        pc[cls][1] += 1
    for cls in sorted(pc):
        h, n = pc[cls]
        tag = '未见类' if cls in UNSEEN else '已见'
        print(f'per-class {cls:14s} n={n:4d} pck={h / n:.4f} [{tag}]')
    # 基线（A0）per-frame 选择，用于找案例（最优臂 vs 基线 A0）
    _, picks_a0 = arm_pck(caches['val'], None, peak_fn)
    best_pk_of = {rec['idx']: pk for rec, pk in zip(caches['val'], picks_of[best_name])}
    a0_pk_of = {rec['idx']: pk for rec, pk in zip(caches['val'], picks_a0)}
    succ, fail = [], []
    for rec in caches['val']:
        b_hit = pck_hit(rec['peaks'][a0_pk_of[rec['idx']]][:2], rec['gt'], rec['obj'])
        s_hit = pck_hit(rec['peaks'][best_pk_of[rec['idx']]][:2], rec['gt'], rec['obj'])
        if (not b_hit) and s_hit and len(succ) < 3:
            succ.append(rec)
        if b_hit and (not s_hit) and len(fail) < 3:
            fail.append(rec)
    render_cases(succ, fail, best_name, best_pk_of, a0_pk_of)
    out = {'meta': {'combos': [list(c) for c in combos], 'alpha': ALPHA,
                    'logistic': 'Adam(lr=1e-3 默认, weight_decay=1e-4), BCE, 2000 steps, seed 0, 全批次',
                    'scales': SCALES, 'n_val': len(caches['val']), 'n_train': len(caches['train'])},
           'g0a': json.load(open(OUT_DIR / 'v3_g0a.json', encoding='utf-8')),
           'logistic_weights': weights,
           'arms': table,
           'train_pck_of_cal_arms': train_pck,
           'best_arm': {'name': best_name, **best,
                        'selection_log': sel_log,
                        'rule': 'overall 降序；±0.0005 并列取 hard 高者'},
           'g1_verdict': g1,
           'n_arms_ge_04643': n_pass,
           'per_class_best': {c: {'hits': h, 'n': n, 'pck': round(h / n, 4),
                                  'unseen': c in UNSEEN} for c, (h, n) in sorted(pc.items())},
           'cases': {'success': [f'{r["inst"]}_f{r["frame_id"]}' for r in succ],
                     'failure': [f'{r["inst"]}_f{r["frame_id"]}' for r in fail]}}
    (OUT_DIR / 'v3_stage2.json').write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                            encoding='utf-8')
    print('落盘: v3_stage2.json + v3_cases/')


def render_cases(succ, fail, best_name, best_pk_of, a0_pk_of):
    """6 张机制案例图（3 miss→hit + 3 hit→miss），风格参照 v_rerank_cases/。"""
    _, ds, model, _, _ = load_model_and_ds('val')
    case_dir = OUT_DIR / 'v3_cases'
    case_dir.mkdir(parents=True, exist_ok=True)
    for tag, items in [('success', succ), ('failure', fail)]:
        for rec in items:
            inst_dir, sj, _ = ds.samples[rec['idx']]
            ir_raw = cv2.cvtColor(cv2.imread(str(sj.with_suffix('.jpg'))), cv2.COLOR_BGR2RGB)
            ir_raw, _, _, _ = _letterbox(ir_raw, IMG_SIZE)
            ir_bgr = cv2.cvtColor(ir_raw, cv2.COLOR_RGB2BGR)
            with torch.no_grad():
                b = ds[rec['idx']]
                logits = model(b['ir'].unsqueeze(0).to(DEVICE),
                               b['templates'].unsqueeze(0).to(DEVICE))
            heat = torch.sigmoid(logits)[0, 0].cpu().numpy()
            hm = (heat - heat.min()) / (heat.max() - heat.min() + 1e-9)
            heat_bgr = cv2.applyColorMap((hm * 255).astype(np.uint8), cv2.COLORMAP_JET)
            disp = cv2.addWeighted(ir_bgr, 0.55, heat_bgr, 0.45, 0)
            cv2.circle(disp, (int(rec['gt'][0]), int(rec['gt'][1])), 10, (0, 200, 0), 2)
            best_pk = best_pk_of[rec['idx']]
            for pi, (x, y, v) in enumerate(rec['peaks']):
                color = (0, 0, 255) if pi == best_pk else (255, 0, 0)
                cv2.drawMarker(disp, (int(x), int(y)), color, cv2.MARKER_CROSS, 18, 2)
                cv2.putText(disp, str(pi), (int(x) + 6, int(y) - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            bp = rec['peaks'][a0_pk_of[rec['idx']]][:2]
            sp = rec['peaks'][best_pk][:2]
            cv2.putText(disp, f'{tag} {best_name} inst={rec["inst"]} f{rec["frame_id"]} d={rec["difficulty"]}',
                        (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            cv2.putText(disp, f'base=({bp[0]:.0f},{bp[1]:.0f}) v3=({sp[0]:.0f},{sp[1]:.0f}) '
                              f'gt=({rec["gt"][0]:.0f},{rec["gt"][1]:.0f})',
                        (8, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            pk = ' '.join(f'{i}:({int(p[0])},{int(p[1])})v{p[2]:.2f}' for i, p in enumerate(rec['peaks']))
            cv2.putText(disp, pk[:95], (8, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            out = case_dir / f'{tag}_{best_name}_{rec["inst"]}_f{rec["frame_id"]}.png'
            cv2.imwrite(str(out), disp)
            print(f'案例 {tag}: {out}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--g0a', action='store_true')
    ap.add_argument('--tmpl', action='store_true')
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--stage1', action='store_true')
    ap.add_argument('--split', default='val', choices=['val', 'train'])
    ap.add_argument('--stage2', action='store_true')
    args = ap.parse_args()
    if args.g0a:
        g0a()
    elif args.tmpl:
        tmpl_cache()
    elif args.smoke:
        smoke()
    elif args.stage1:
        stage1(args.split)
    elif args.stage2:
        stage2()
    else:
        ap.print_help()


if __name__ == '__main__':
    main()
