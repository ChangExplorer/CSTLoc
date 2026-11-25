# 实验 V：判别验证器重排——UCM 专家从注入器改为后验验证器（纯推理，零训练）
#
# 对 val 每一帧（n=1736）：
#   ① j2_expert 前向得热图，argmax 预测 = 基线（PCK_before）；
#   ② top-K 峰（K=5，最小间距 32px，峰值降序，迭代 NMS）；
#   ③ 每峰从 letterboxed IR（384×640）裁 224×224（越界补零）→ DINOv2 intermediate_gap
#      → expert(head_v0.pt) → 256-d 候选嵌入；
#   ④ 模板嵌入：每 val 实例全部 RGB 模板切片过同管线取均值（缓存 v_template_emb.pt）；
#   ⑤ score_final = (1−α)·峰值(sigmoid 热图值) + α·cos_sim(候选, 模板)，α ∈ {0, 0.5, 1.0}；
#      rank-1 峰 = 该 α 预测。α=0 = 基线（G0 锚）。
# 已知局限（报告注明）：固定 224 裁窗不含目标尺度先验。
#
# 落盘：v_rerank_val.json（逐帧前后预测 + 各 α PCK）、v_template_emb.pt、v_rerank_cases/。
# 用法: CSTLOC_IR_ENCODER=infmae python tools/v_verifier_rerank.py
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
from data.cstloc_dataset import CSTLocDataset, _letterbox  # noqa: E402
from models.localizer import Localizer  # noqa: E402
from models.encoders.dinov2_encoder import DINOv2TemplateEncoder  # noqa: E402
from models.identity_expert import IdentityExpert  # noqa: E402

ROOT = Path(os.environ.get('CSTLOC_DATA_ROOT', '/home/user/Code/Datasets/CSTLoc'))
EXPERT_CKPT = CODE / 'checkpoints' / 'j2_expert_pretrain' / 'head_v0.pt'
MODEL_CKPT = CODE / 'checkpoints' / 'j2_expert' / 'best_model.pth'
OUT_DIR = CODE / 'diag_out'
CASE_DIR = OUT_DIR / 'v_rerank_cases'
DEVICE = 'cuda'
IMG_SIZE = (384, 640)
PATCH = 224
K = 5
MIN_DIST = 32
ALPHAS = [0.0, 0.5, 1.0]
PCK_THRESH = 0.5
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class _Idx:
    def __init__(self, ds):
        self.ds = ds

    def __getitem__(self, idx):
        s = self.ds[idx]
        s['_idx'] = idx
        return s

    def __len__(self):
        return len(self.ds)

    @property
    def samples(self):
        return self.ds.samples


def nms_peaks(heat_np, k=K, min_dist=MIN_DIST):
    """heat_np (H,W) 0-1。迭代取全局最大并抑制 min_dist 半径圆盘。返回 [(x,y,val)] 降序。"""
    h = heat_np.copy()
    H, W = h.shape
    yy, xx = np.ogrid[:H, :W]
    peaks = []
    for _ in range(k):
        idx = int(np.argmax(h))
        y, x = divmod(idx, W)
        v = float(h[y, x])
        if v <= 0:
            break
        peaks.append((x, y, v))
        h[(xx - x) ** 2 + (yy - y) ** 2 < min_dist ** 2] = 0.0
    return peaks


def crop_patch(ir_raw_bgr, cx, cy):
    """从 letterboxed IR（BGR uint8）裁 224×224，越界补零。返回 RGB uint8 (224,224,3)。"""
    H, W = ir_raw_bgr.shape[:2]
    hw = PATCH // 2
    x0, y0 = int(cx) - hw, int(cy) - hw
    x1, y1 = int(cx) + hw, int(cy) + hw
    x0c, y0c = max(0, x0), max(0, y0)
    x1c, y1c = min(W, x1), min(H, y1)
    patch = np.zeros((PATCH, PATCH, 3), dtype=np.uint8)
    if x1c > x0c and y1c > y0c:
        patch[y0c - y0:y1c - y0, x0c - x0:x1c - x0] = ir_raw_bgr[y0c:y1c, x0c:x1c]
    return cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)


def norm224(img_rgb):
    """RGB uint8 224×224 → CHW ImageNet 归一化（与 _to_tensor 同口径）。"""
    t = torch.from_numpy(img_rgb).float().permute(2, 0, 1) / 255.0
    return (t - MEAN) / STD



def _norm_fusion_pred(fr, alpha):
    """归一化融合（整改口径）：score = (1−α)·peak_norm + α·cos_raw；
    peak_norm = 该帧 K 峰 sigmoid 值的帧内 min-max；cos_raw 保留绝对标定。"""
    vals = [fr['scores'][str(pi)]['peak_val'] for pi in range(len(fr['peaks']))]
    lo, hi = min(vals), max(vals)
    span = hi - lo
    best_pi, best_s = -1, -1e9
    for pi in range(len(fr['peaks'])):
        s = fr['scores'][str(pi)]
        pn = (s['peak_val'] - lo) / span if span > 1e-9 else 0.0
        sc = (1 - alpha) * pn + alpha * s['cos']
        if sc > best_s:
            best_s, best_pi = sc, pi
    return (fr['peaks'][best_pi][0], fr['peaks'][best_pi][1]), best_pi


def normalized_scan():
    """V 整改：归一化融合 α 曲线（0~1.0 步 0.05）→ v_rerank_normalized.json（新文件）。
    数据全部来自 v_rerank_val.json 逐帧（peak_val/cos），无需重跑模型。"""
    doc = json.load(open(OUT_DIR / 'v_rerank_val.json', encoding='utf-8'))
    frames = doc['frames']
    alphas = [round(a, 2) for a in np.arange(0, 1.0001, 0.05)]

    def pck(pred, fr):
        return np.hypot(pred[0] - fr['gt'][0], pred[1] - fr['gt'][1]) <= \
            PCK_THRESH * max(fr['obj'][0], fr['obj'][1])

    curve = {}
    for a in alphas:
        agg = {'overall': [0, 0], 'easy': [0, 0], 'hard': [0, 0]}
        for fr in frames:
            pred, _ = _norm_fusion_pred(fr, a)
            hit = pck(pred, fr)
            agg['overall'][0] += int(hit); agg['overall'][1] += 1
            if fr['difficulty'] in ('easy', 'hard'):
                agg[fr['difficulty']][0] += int(hit); agg[fr['difficulty']][1] += 1
        curve[str(a)] = {k: round(v[0] / v[1], 4) if v[1] else None for k, v in agg.items()}
    # oracle top-5（候选集含正确峰比例）
    oracle_hit = sum(1 for fr in frames
                     if any(pck((p[0], p[1]), fr) for p in fr['peaks']))
    oracle = round(oracle_hit / len(frames), 4)
    # 随机选峰基线（固定种子）
    rng = np.random.RandomState(0)
    rand_hit = sum(1 for fr in frames
                   if pck(tuple(fr['peaks'][int(rng.randint(len(fr['peaks'])))][:2]), fr))
    rand_pck = round(rand_hit / len(frames), 4)
    # α=0.8 per-class
    pc = {}
    for fr in frames:
        cls = fr['inst'].split('_')[0] + '_'
        pc.setdefault(cls, [0, 0])
        pred, _ = _norm_fusion_pred(fr, 0.8)
        pc[cls][0] += int(pck(pred, fr)); pc[cls][1] += 1
    out = {'fusion': 'score=(1-a)*peak_norm(frame-wise minmax) + a*cos_raw',
           'note': 'V 整改：量纲伪象修正（原实现 peak_val 未归一化，α=0.5 实为纯 cos）',
           'n_frames': len(frames),
           'curve': curve,
           'a_08': {'overall': curve['0.8']['overall'], 'easy': curve['0.8']['easy'],
                    'hard': curve['0.8']['hard']},
           'baseline_overall': doc['g0_overall_pck'],
           'oracle_top5': oracle, 'random_peak_pck': rand_pck,
           'per_class_a08': {c: {'hits': h, 'n': n, 'pck': round(h / n, 4)} for c, (h, n) in pc.items()}}
    (OUT_DIR / 'v_rerank_normalized.json').write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    for a in (0.0, 0.5, 0.75, 0.8, 0.9, 1.0):
        print(f"α={a}: overall={curve[str(a)]['overall']} easy={curve[str(a)]['easy']} "
              f"hard={curve[str(a)]['hard']}")
    print(f'oracle top-5 = {oracle} | 随机选峰 = {rand_pck} | α=0.8 per-class 见 JSON')
    return frames


def cases_only():
    """--cases-only：从 v_rerank_val.json 重算 α=0.5（重排实际发生的档）的成败案例并渲染。
    因最优 α*=0.0（无重排），案例标注为机制层（α=0.5）说明重排为何净负。"""
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--cases-only', action='store_true')
    a_cases = 0.8
    doc = json.load(open(OUT_DIR / 'v_rerank_val.json', encoding='utf-8'))
    cfg = yaml.safe_load(open(CODE / 'configs' / 'v2_s1_seed42.yaml', encoding='utf-8'))
    ds = CSTLocDataset(root=str(ROOT), split='val', mode='test',
                       img_size=tuple(cfg['img_size']), patch_size=cfg['patch_size'],
                       sigma_factor=cfg['sigma_factor'], n_templates=cfg['eval_n_templates'],
                       min_gap_frames=cfg['min_gap_frames'], instances=None)

    def pck(pred, fr):
        return np.hypot(pred[0] - fr['gt'][0], pred[1] - fr['gt'][1]) <= \
            PCK_THRESH * max(fr['obj'][0], fr['obj'][1])

    model = Localizer(encoder='dinov2', modulator='correlation', pretrained=True,
                      pretrained_path=cfg.get('pretrained_path'), ir_encoder='infmae',
                      use_id=True, identity_expert=str(EXPERT_CKPT), mask_null_p=0.0).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_CKPT, map_location='cpu'), strict=True)
    model.eval()
    succ, fail = [], []
    for fr in doc['frames']:
        pred, best_pi = _norm_fusion_pred(fr, a_cases)
        b_hit, s_hit = pck(fr['baseline_pred'], fr), pck(pred, fr)
        if (not b_hit) and s_hit and len(succ) < 3:
            succ.append((fr, pred, best_pi))
        if b_hit and (not s_hit) and len(fail) < 3:
            fail.append((fr, pred, best_pi))
    CASE_DIR.mkdir(parents=True, exist_ok=True)
    for tag, items in [('success_a08', succ), ('failure_a08', fail)]:
        for fr, pred, best_pi in items:
            idx = fr['idx']
            inst_dir, sj, _ = ds.samples[idx]
            ir_raw = cv2.cvtColor(cv2.imread(str(sj.with_suffix('.jpg'))), cv2.COLOR_BGR2RGB)
            ir_raw, _, _, _ = _letterbox(ir_raw, IMG_SIZE)
            ir_bgr = cv2.cvtColor(ir_raw, cv2.COLOR_RGB2BGR)
            b = ds[idx]
            with torch.no_grad():
                logits = model(b['ir'].unsqueeze(0).to(DEVICE),
                               b['templates'].unsqueeze(0).to(DEVICE))
            heat = torch.sigmoid(logits)[0, 0].cpu().numpy()
            hm = (heat - heat.min()) / (heat.max() - heat.min() + 1e-9)
            heat_bgr = cv2.applyColorMap((hm * 255).astype(np.uint8), cv2.COLORMAP_JET)
            disp = cv2.addWeighted(ir_bgr, 0.55, heat_bgr, 0.45, 0)
            cv2.circle(disp, (int(fr['gt'][0]), int(fr['gt'][1])), 10, (0, 200, 0), 2)
            for pi, (x, y, v) in enumerate(fr['peaks']):
                color = (0, 0, 255) if pi == best_pi else (255, 0, 0)
                cv2.drawMarker(disp, (int(x), int(y)), color, cv2.MARKER_CROSS, 18, 2)
                cv2.putText(disp, str(pi), (int(x) + 6, int(y) - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            bp = fr['baseline_pred']
            cv2.putText(disp, f'{tag} inst={fr["inst"]} f{fr["frame_id"]} d={fr["difficulty"]} (α={a_cases})',
                        (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            cv2.putText(disp, f'baseline=({bp[0]:.0f},{bp[1]:.0f}) rerank=({pred[0]:.0f},{pred[1]:.0f}) '
                              f'gt=({fr["gt"][0]:.0f},{fr["gt"][1]:.0f})',
                        (8, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            rank_txt = ' '.join(f'{pi}:v{s["peak_val"]:.2f}c{s["cos"]:+.2f}'
                                for pi, s in sorted(fr['scores'].items()))
            cv2.putText(disp, rank_txt[:95], (8, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            out = CASE_DIR / f'{tag}_{fr["inst"]}_f{fr["frame_id"]}.png'
            cv2.imwrite(str(out), disp)
            print(f'机制案例 {tag}: {out}')


def main():
    torch.manual_seed(0)
    cfg = yaml.safe_load(open(CODE / 'configs' / 'v2_s1_seed42.yaml', encoding='utf-8'))
    eval_cfg = dict(cfg, n_templates=cfg['eval_n_templates'])
    ds = _Idx(CSTLocDataset(root=str(ROOT), split='val', mode='test',
                            img_size=tuple(cfg['img_size']), patch_size=cfg['patch_size'],
                            sigma_factor=cfg['sigma_factor'], n_templates=cfg['eval_n_templates'],
                            min_gap_frames=cfg['min_gap_frames'], instances=None))
    loader = DataLoader(ds, batch_size=12, shuffle=False, num_workers=4, pin_memory=True)
    print(f'val 样本数: {len(ds)}')

    model = Localizer(encoder='dinov2', modulator='correlation', pretrained=True,
                      pretrained_path=cfg.get('pretrained_path'), ir_encoder='infmae',
                      use_id=True, identity_expert=str(EXPERT_CKPT), mask_null_p=0.0).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_CKPT, map_location='cpu'), strict=True)
    model.eval()
    enc = DINOv2TemplateEncoder(pretrained=True).cuda().eval()
    expert = IdentityExpert(str(EXPERT_CKPT)).cuda().eval()
    print(f'模型加载: {MODEL_CKPT} | 专家: {EXPERT_CKPT}')

    # ---- ④ 模板嵌入（val 实例全部模板切片均值，缓存一次）----
    tmpl_cache = OUT_DIR / 'v_template_emb.pt'
    inst_dirs = sorted(p.name for p in (ROOT / 'val').iterdir() if p.is_dir())
    if tmpl_cache.exists():
        tmpl_emb = torch.load(tmpl_cache, map_location='cpu')
        print(f'模板嵌入缓存命中: {tmpl_cache}')
    else:
        tmpl_emb = {}
        todo = []
        for inst in inst_dirs:
            tdir = ROOT / 'val' / inst / 'rgb' / 'template'
            for p in sorted(tdir.glob('*.jpg'), key=lambda q: int(q.stem)):
                img = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (PATCH, PATCH), interpolation=cv2.INTER_LINEAR)
                todo.append((inst, norm224(img)))
        print(f'模板切片总数: {len(todo)}，过 GAP+专家...')
        embs = {i: [] for i in inst_dirs}
        with torch.no_grad():
            for s in range(0, len(todo), 64):
                xs = torch.stack([t[1] for t in todo[s:s + 64]]).to(DEVICE)
                gap = enc.intermediate_gap(xs)
                e = expert(gap).cpu()
                for j, (inst, _) in enumerate(todo[s:s + 64]):
                    embs[inst].append(e[j])
        for inst in inst_dirs:
            tmpl_emb[inst] = torch.stack(embs[inst]).mean(0)   # (256,)
        torch.save(tmpl_emb, tmpl_cache)
        print(f'模板嵌入落盘: {tmpl_cache}')

    # ---- 逐帧：热图 + 峰 + 候选嵌入 + α 重排 ----
    frames = []          # 每帧记录
    crop_todo = []       # (frame_idx, peak_idx, patch_rgb)
    heatmaps = {}        # frame_idx → (heat_np, ir_raw_bgr)（仅存用于案例可视化的帧，先全存 float16）
    baseline = {}        # frame_idx → argmax pred
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            ir = batch['ir'].to(DEVICE)
            templates = batch['templates'].to(DEVICE)
            gt = batch['gt_center']
            size = batch['obj_size']
            logits = model(ir, templates)
            heat = torch.sigmoid(logits)[:, 0].float()        # (B,384,640)
            for i in range(ir.size(0)):
                idx = int(batch['_idx'][i])
                inst = batch['instance'][i]
                diff = batch['difficulty'][i]
                hn = heat[i].cpu().numpy()
                argmax_yx = np.unravel_index(int(np.argmax(hn)), hn.shape)
                baseline[idx] = (float(argmax_yx[1]), float(argmax_yx[0]))
                peaks = nms_peaks(hn, K, MIN_DIST)
                frames.append({'idx': idx, 'inst': inst, 'difficulty': diff,
                               'gt': [float(gt[i][0]), float(gt[i][1])],
                               'obj': [float(size[i][0]), float(size[i][1])],
                               'baseline_pred': baseline[idx], 'peaks': peaks,
                               'frame_id': int(ds.samples[idx][1].stem)})
            if bi % 20 == 0:
                print(f'  热图 {bi * 12}/{len(ds)}')
    # 候选裁窗（重读 raw IR）
    for fr in frames:
        idx = fr['idx']
        inst_dir, sj, _ = ds.samples[idx]
        ir_raw = cv2.cvtColor(cv2.imread(str(sj.with_suffix('.jpg'))), cv2.COLOR_BGR2RGB)
        ir_raw, _, _, _ = _letterbox(ir_raw, IMG_SIZE)
        ir_raw_bgr = cv2.cvtColor(ir_raw, cv2.COLOR_RGB2BGR)
        fr['_raw'] = ir_raw_bgr
        for pi, (x, y, v) in enumerate(fr['peaks']):
            crop_todo.append((fr, pi, crop_patch(ir_raw_bgr, x, y)))
    print(f'候选裁窗总数: {len(crop_todo)}，过 GAP+专家...')
    with torch.no_grad():
        for s in range(0, len(crop_todo), 128):
            xs = torch.stack([norm224(c) for _, _, c in crop_todo[s:s + 128]]).to(DEVICE)
            gap = enc.intermediate_gap(xs)
            e = expert(gap).cpu()
            for j, (fr, pi, _) in enumerate(crop_todo[s:s + 128]):
                fr.setdefault('cand_emb', {})[pi] = e[j]
    # 释放 raw（案例可视化时再读）
    for fr in frames:
        fr.pop('_raw', None)

    # ---- ⑤ α 融合重排 + PCK ----
    def pck(pred, fr):
        dist = np.hypot(pred[0] - fr['gt'][0], pred[1] - fr['gt'][1])
        return dist <= PCK_THRESH * max(fr['obj'][0], fr['obj'][1])

    res = {a: {'overall': [0, 0], 'easy': [0, 0], 'hard': [0, 0], 'preds': {}} for a in ALPHAS}
    for fr in frames:
        tmpl_e = tmpl_emb[fr['inst']]
        scores = {}
        for pi, (x, y, v) in enumerate(fr['peaks']):
            ce = fr['cand_emb'][pi]
            cos = float(F.cosine_similarity(ce.unsqueeze(0), tmpl_e.unsqueeze(0)).item())
            scores[pi] = {'peak_val': v, 'cos': cos}
        fr['scores'] = scores
        for a in ALPHAS:
            best_pi = max(scores, key=lambda pi: (1 - a) * scores[pi]['peak_val'] + a * scores[pi]['cos'])
            pred = (fr['peaks'][best_pi][0], fr['peaks'][best_pi][1])
            res[a]['preds'][fr['idx']] = {'pred': pred, 'peak_idx': best_pi,
                                          'peak_val': scores[best_pi]['peak_val'],
                                          'cos': scores[best_pi]['cos']}
            hit = pck(pred, fr)
            res[a]['overall'][0] += int(hit)
            res[a]['overall'][1] += 1
            if fr['difficulty'] in ('easy', 'hard'):
                res[a][fr['difficulty']][0] += int(hit)
                res[a][fr['difficulty']][1] += 1

    # G0 锚
    g0 = res[0.0]['overall'][0] / res[0.0]['overall'][1]
    print(f'G0: α=0 overall PCK = {g0:.4f}（锚 0.4412）')
    for a in ALPHAS:
        o = res[a]['overall'][0] / res[a]['overall'][1]
        e = res[a]['easy'][0] / res[a]['easy'][1] if res[a]['easy'][1] else None
        h = res[a]['hard'][0] / res[a]['hard'][1] if res[a]['hard'][1] else None
        print(f'  α={a}: overall={o:.4f} easy={e} hard={h}')

    # per-class（最优 α*）
    a_star = max(ALPHAS, key=lambda a: res[a]['overall'][0] / res[a]['overall'][1])
    print(f'最优 α* = {a_star}')
    per_class = {}
    for fr in frames:
        cls = fr['inst'].split('_')[0] + '_'
        per_class.setdefault(cls, [0, 0])
        hit = pck(res[a_star]['preds'][fr['idx']]['pred'], fr)
        per_class[cls][0] += int(hit)
        per_class[cls][1] += 1

    # 案例：success = 基线 miss → α* hit；failure = 基线 hit → α* miss（按帧序前 3）
    succ, fail = [], []
    for fr in frames:
        b_hit = pck(fr['baseline_pred'], fr)
        s_hit = pck(res[a_star]['preds'][fr['idx']]['pred'], fr)
        if (not b_hit) and s_hit and len(succ) < 3:
            succ.append(fr['idx'])
        if b_hit and (not s_hit) and len(fail) < 3:
            fail.append(fr['idx'])
    # 案例可视化
    CASE_DIR.mkdir(parents=True, exist_ok=True)
    for tag, idxs in [('success', succ), ('failure', fail)]:
        for fr in [f for f in frames if f['idx'] in idxs]:
            idx = fr['idx']
            inst_dir, sj, _ = ds.samples[idx]
            ir_raw = cv2.cvtColor(cv2.imread(str(sj.with_suffix('.jpg'))), cv2.COLOR_BGR2RGB)
            ir_raw, _, _, _ = _letterbox(ir_raw, IMG_SIZE)
            ir_bgr = cv2.cvtColor(ir_raw, cv2.COLOR_RGB2BGR)
            # 热图
            with torch.no_grad():
                b = ds[idx]
                logits = model(b['ir'].unsqueeze(0).to(DEVICE),
                               b['templates'].unsqueeze(0).to(DEVICE))
            heat = torch.sigmoid(logits)[0, 0].cpu().numpy()
            hm = (heat - heat.min()) / (heat.max() - heat.min() + 1e-9)
            heat_bgr = cv2.applyColorMap((hm * 255).astype(np.uint8), cv2.COLORMAP_JET)
            disp = cv2.addWeighted(ir_bgr, 0.55, heat_bgr, 0.45, 0)
            cv2.circle(disp, (int(fr['gt'][0]), int(fr['gt'][1])), 10, (0, 200, 0), 2)
            for pi, (x, y, v) in enumerate(fr['peaks']):
                color = (0, 0, 255) if pi == res[a_star]['preds'][idx]['peak_idx'] else (255, 0, 0)
                cv2.drawMarker(disp, (int(x), int(y)), color, cv2.MARKER_CROSS, 18, 2)
                cv2.putText(disp, str(pi), (int(x) + 6, int(y) - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            bpred, spred = fr['baseline_pred'], res[a_star]['preds'][idx]['pred']
            cv2.putText(disp, f'{tag} inst={fr["inst"]} f{fr["frame_id"]} d={fr["difficulty"]}',
                        (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(disp, f'baseline=({bpred[0]:.0f},{bpred[1]:.0f}) '
                              f'alpha*={a_star}=({spred[0]:.0f},{spred[1]:.0f}) gt=({fr["gt"][0]:.0f},{fr["gt"][1]:.0f})',
                        (8, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            rank_txt = ' '.join(f'{pi}:v{s["peak_val"]:.2f}c{s["cos"]:+.2f}'
                                for pi, s in sorted(fr['scores'].items()))
            cv2.putText(disp, rank_txt[:95], (8, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            out = CASE_DIR / f'{tag}_{fr["inst"]}_f{fr["frame_id"]}.png'
            cv2.imwrite(str(out), disp)
            print(f'案例 {tag}: {out}')
    # 落盘
    doc = {'meta': {'model': str(MODEL_CKPT), 'expert': str(EXPERT_CKPT),
                    'k': K, 'min_dist': MIN_DIST, 'alphas': ALPHAS,
                    'patch': PATCH, 'a_star': a_star,
                    'n_frames': len(frames),
                    'pck_def': '0.5*max(w,h)', 'difficulty_buckets': 'overall/easy/hard'},
           'g0_overall_pck': round(g0, 4),
           'pck_by_alpha': {str(a): {'overall': round(res[a]['overall'][0] / res[a]['overall'][1], 4),
                                     'easy': (round(res[a]['easy'][0] / res[a]['easy'][1], 4)
                                              if res[a]['easy'][1] else None),
                                     'hard': (round(res[a]['hard'][0] / res[a]['hard'][1], 4)
                                              if res[a]['hard'][1] else None)}
                            for a in ALPHAS},
           'per_class_alpha_star': {c: {'hits': h, 'n': n, 'pck': round(h / n, 4)} for c, (h, n) in per_class.items()},
           'frames': [{'idx': f['idx'], 'inst': f['inst'], 'frame_id': f['frame_id'],
                       'difficulty': f['difficulty'], 'gt': f['gt'], 'obj': f['obj'],
                       'baseline_pred': f['baseline_pred'],
                       'peaks': f['peaks'], 'scores': f['scores'],
                       'pred_alpha_star': res[a_star]['preds'][f['idx']]['pred'],
                       'pck_before': bool(pck(f['baseline_pred'], f)),
                       'pck_after': bool(pck(res[a_star]['preds'][f['idx']]['pred'], f))}
                      for f in frames]}
    (OUT_DIR / 'v_rerank_val.json').write_text(
        json.dumps(doc, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'落盘: v_rerank_val.json（{len(frames)} 帧）+ v_template_emb.pt + v_rerank_cases/（{len(succ)}+{len(fail)}）')


if __name__ == '__main__':
    import sys as _sys
    if '--cases-only' in _sys.argv:
        cases_only()
    elif '--normalized-scan' in _sys.argv:
        normalized_scan()
    else:
        main()
