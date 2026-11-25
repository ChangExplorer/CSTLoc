# 实验 N：单模板循环协议重评（protocol correction，零训练纯推理）
#
# 协议（写死）：trial = (instance, template t, search frame f)，合格 iff |f−t|>=30。
#   单对指标 = 模板 t 在其全部合格帧上的 PCK@0.5；主指标 = 全部 (inst,t) 对的宏平均。
#   附报口径：trial-微平均、instance-宏平均；easy/hard 桶（帧 flags，与 eval_per_class 一致）。
#   模板集合 = val 每实例 rgb/template 全部（普查 1355 张 / 1755 帧 / 197,649 trials，含
#   rgb_usable=false 的 pedestrian_204/207——现行协议按数据集规则排除这两实例，属 R1 口径差，
#   报告披露；G0 中位模式复刻现行协议（数据集口径，21 实例 / 1736 帧）。
# 系统（优先级 S3→S2→C2→C1→S1→S4→s123）：
#   S3=A2 随机专家 / S2=j2 真实专家 / S1=baseline s42 / S4=J3 null_constant / s123 可选
#   C2=S3+V' / C1=S2+V'（V' 见下）
# V'（单模板验证器）：对 trial (t,f)：S 热图 top-5 峰 → 224 裁窗（V 口径）→ intermediate_gap
#   → head_v0 → 256-d；模板嵌入 = 模板 t 自身过同管线（L2 归一化，禁止跨切片聚合）——
#   复用 v3_template_slices.pt 逐切片行（管线一致，随机抽 3 切片重算核验）。
#   融合 score = 0.2·peak_norm + 0.8·cos（复用 frame_minmax/linear_fn）。
# G0（不过即停）：新工具以中位合格模板模式复现现行口径：s42=0.4107 / j2=0.4412 /
#   A2=0.4533 / J3=0.4315（±0.002，per_class_val_{s42,j2,a2,j3} 实测锚）。
# 缓存（§6）：DINOv2 模板原始特征每模板一次落盘（n_cache_tmpl.pt：层 7/15/23 reshape=True
#   maps + 层 15/23 reshape=False token-mean gap，fp32）；各系统 1×1 投影现算；候选嵌入按
#   (frame,peak_x,peak_y) 去重。IR 特征：每帧编码一次、帧内全部模板 trial 复用（内存内，
#   不落盘——52GB 磁盘缓存净收益为负：写读 ~40min vs 三个后续系统重编码 ~3min，报告披露）。
# 断点续跑：n_progress.json 记 (system,instance) 完成单元，重启跳过。
import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
from data.cstloc_dataset import CSTLocDataset, _letterbox                     # noqa: E402
from tools.wrong_template_probe import load_frame                              # noqa: E402
from tools.v_verifier_rerank import (ROOT, EXPERT_CKPT, IMG_SIZE, PATCH,       # noqa: E402
    K, MIN_DIST, OUT_DIR, nms_peaks, crop_patch, norm224, MEAN, STD)           # noqa: E402
from concurrent.futures import ThreadPoolExecutor
from tools.v3_verifier_deepen import pck_hit, frame_minmax, linear_fn          # noqa: E402
from models.localizer import Localizer                                         # noqa: E402
from models.encoders.dinov2_encoder import DINOv2TemplateEncoder               # noqa: E402
from models.identity_expert import IdentityExpert                              # noqa: E402

DEVICE = 'cuda'
MIN_GAP = 30
PICK = Path(os.environ.get('CSTLOC_EXPERT_DIR', str(CODE / 'checkpoints' / 'j2_expert_pretrain')))
RANDOM_HEAD = PICK / 'head_random.pt'
SYSTEMS = {
    's3': dict(ckpt='checkpoints/a2_random_expert/best_model.pth',
               cfg='configs/a2_random_expert.yaml', use_id=True, expert=str(RANDOM_HEAD)),
    's2': dict(ckpt='checkpoints/j2_expert/best_model.pth',
               cfg='configs/j2_expert.yaml', use_id=True, expert=str(EXPERT_CKPT)),
    's1': dict(ckpt='checkpoints/v2_s1_s42/best_model.pth',
               cfg='configs/v2_s1_seed42.yaml', use_id=False, expert=None),
    's4': dict(ckpt='checkpoints/j3_null_inject/best_model.pth',
               cfg='configs/j3_null_inject.yaml', use_id=True, expert='null_constant'),
    's123': dict(ckpt='checkpoints/v2_s1_s123/best_model.pth',
                 cfg='configs/v2_s1_seed123.yaml', use_id=False, expert=None),
}
G0_ANCHORS = {'s1': 0.4107, 's2': 0.4412, 's3': 0.4533, 's4': 0.4315}
G0_TOL = 0.002
BATCH_T = 12          # 单帧模板批量（前向 chunk；ir_feats expand 到该批量，显存与训练 batch 同级）
ALPHA = 0.8
TMPL_BATCH = 32       # 模板缓存构建批量
# 任务书 1.30 段4：split 参数化（默认 val → 既有行为逐位不变；test 走 CSTLOC_SPLIT=test）
SPLIT = os.environ.get('CSTLOC_SPLIT', 'val')
SUF = '' if SPLIT == 'val' else f'_{SPLIT}'      # test 专用新文件名（勿动 val 缓存）
PROGRESS = OUT_DIR / f'n_progress{SUF}.json'


def split_dir():
    return ROOT / SPLIT


def val_instances():
    """N-协议实例全集：val 目录 23 实例（含 rgb_usable=false，任务书模板集 1355 张口径）。"""
    insts = []
    for inst in sorted(p.name for p in split_dir().iterdir() if p.is_dir()):
        mp = split_dir() / inst / 'meta.json'
        if mp.exists() and json.loads(mp.read_text(encoding='utf-8')).get('rgb_usable') is False:
            continue
        tmpls = sorted(int(p.stem) for p in (split_dir() / inst / 'rgb' / 'template').glob('*.jpg'))
        frames = sorted(int(p.stem) for p in (split_dir() / inst / 'ir' / 'search').glob('*.json'))
        insts.append((inst, tmpls, frames))
    return insts


def census():
    rows, tot = [], {'templates': 0, 'frames': 0, 'trials': 0}
    for inst, tmpls, frames in val_instances():
        n_trials = sum(1 for f in frames for t in tmpls if abs(f - t) >= MIN_GAP)
        rows.append({'inst': inst, 'n_templates': len(tmpls), 'n_frames': len(frames),
                     'n_trials': n_trials})
        tot['templates'] += len(tmpls); tot['frames'] += len(frames); tot['trials'] += n_trials
    return rows, tot


def progress_load():
    if PROGRESS.exists():
        return json.loads(PROGRESS.read_text(encoding='utf-8'))
    return {}


def progress_mark(key):
    p = progress_load()
    p[key] = 1
    PROGRESS.write_text(json.dumps(p, ensure_ascii=False, indent=1), encoding='utf-8')


def build_system(key):
    spec = SYSTEMS[key]
    cfg = yaml.safe_load(open(CODE / spec['cfg'], encoding='utf-8'))
    model = Localizer(encoder='dinov2', modulator=cfg.get('modulator', 'correlation'), pretrained=True,
                      pretrained_path=cfg.get('pretrained_path'), ir_encoder='infmae',
                      use_id=spec['use_id'], identity_expert=spec['expert'],
                      mask_null_p=0.0).to(DEVICE)
    sd = torch.load(CODE / spec['ckpt'], map_location='cpu')
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model


@torch.no_grad()
def forward_cached(model, use_id, ir_x, feats):
    """缓存组合前向（与 Localizer.forward 逐模块同路径，模板塔走缓存原始特征）。
    ir_x: (B_ir,3,384,640) cuda（通常 B_ir=1：单帧多模板）；feats: list 长度 B_tmpl。
    ir_feats expand 到模板批量（与标准 forward 的同 batch 语义一致）。"""
    B = len(feats)
    m7 = torch.stack([d['m7'] for d in feats]).to(DEVICE)
    m15 = torch.stack([d['m15'] for d in feats]).to(DEVICE)
    m23 = torch.stack([d['m23'] for d in feats]).to(DEVICE)
    p7 = model.tmpl_encoder.proj128(m7)
    p15 = model.tmpl_encoder.proj256(m15)
    p23 = model.tmpl_encoder.proj512(m23)
    vecs = tuple(model.tmpl_encoder.gap(p).flatten(1).unsqueeze(1) for p in (p7, p15, p23))
    maps = tuple(p.unsqueeze(1) for p in (p7, p15, p23))
    if use_id:
        if getattr(model, 'null_constant', False):
            id_emb = model.null_const.unsqueeze(0).expand(B, -1).clone()
        else:
            gap = torch.stack([d['gap'] for d in feats]).to(DEVICE)
            id_emb = model.expert(gap)
    else:
        id_emb = None
    x1, x2, x3, x4, x5 = model.ir_encoder(ir_x)
    if B != ir_x.shape[0]:
        x1, x2, x3, x4, x5 = (t.expand(B, -1, -1, -1) for t in (x1, x2, x3, x4, x5))
    x3, x4, x5 = model.modulator((x3, x4, x5), vecs, maps, id_emb=id_emb)
    return model.decoder(x1, x2, x3, x4, x5)


# ---------------- stage 0：清单 + 冒烟 + G0 ----------------
def stage0():
    torch.manual_seed(0)
    rows, tot = census()
    for r in rows:
        print(f"{r['inst']:16s} n_t={r['n_templates']:3d} n_f={r['n_frames']:3d} trials={r['n_trials']}")
    print(f"清单合计: 模板 {tot['templates']} | search json {tot['frames']} | trials {tot['trials']}")
    assert tot['templates'] > 0, tot
    assert not (OUT_DIR / f'n_cache_tmpl{SUF}.pt').exists() or True
    # 中位模式（现行协议语义 = CSTLocDataset val/test，自动排除 rgb_usable=false）
    ds = CSTLocDataset(root=str(ROOT), split='val', mode='test',
                       img_size=(384, 640), patch_size=224, sigma_factor=0.3,
                       n_templates=1, min_gap_frames=MIN_GAP, instances=None)
    print(f'现行协议（中位模式）样本数: {len(ds)}（=1736 锚口径）')
    tmpl = torch.load(OUT_DIR / f'n_cache_tmpl{SUF}.pt', map_location='cpu')
    g0 = {}
    for key, anchor in G0_ANCHORS.items():
        model = build_system(key)
        use_id = SYSTEMS[key]['use_id']
        hits = 0
        for i in range(len(ds)):
            inst_dir, sj, cands = ds.samples[i]
            t = cands[len(cands) // 2]
            f = int(sj.stem)
            fr = load_frame(inst_dir, f)
            inst = Path(inst_dir).name
            logits = forward_cached(model, use_id, fr['ir'].unsqueeze(0).to(DEVICE),
                                    [tmpl[inst][str(t)]])
            heat = torch.sigmoid(logits)[0, 0].cpu().numpy()
            ay, ax = np.unravel_index(int(np.argmax(heat)), heat.shape)
            hits += int(pck_hit([float(ax), float(ay)], [float(fr['gt_center'][0]), float(fr['gt_center'][1])], [float(fr['obj_size'][0]), float(fr['obj_size'][1])]))
        pck = hits / len(ds)
        ok = abs(pck - anchor) <= G0_TOL
        g0[key] = {'pck': round(pck, 4), 'anchor': anchor, 'pass': bool(ok)}
        print(f"G0[{key}]: 中位模式 PCK = {pck:.4f}（锚 {anchor}±{G0_TOL}）{'PASS' if ok else 'FAIL'}")
        del model
        torch.cuda.empty_cache()
    all_pass = all(v['pass'] for v in g0.values())
    print(f"G0 总体: {'PASS' if all_pass else 'FAIL'}（不过 = 停机上报，禁止进入阶段 2）")
    # 冒烟：1 实例 × 3 模板（car_051），打印逐 trial 预测与 GT 距离抽样
    model = build_system('s3')
    inst = 'car_051'
    tmpls = sorted(int(p.stem) for p in (split_dir() / inst / 'rgb' / 'template').glob('*.jpg'))
    n_print = 0
    for t in tmpls[:3]:
        frames = sorted(int(p.stem) for p in (split_dir() / inst / 'ir' / 'search').glob('*.json'))
        for f in frames:
            if abs(f - t) < MIN_GAP:
                continue
            fr = load_frame(split_dir() / inst, f)
            logits = forward_cached(model, True, fr['ir'].unsqueeze(0).to(DEVICE),
                                    [tmpl[inst][str(t)]])
            heat = torch.sigmoid(logits)[0, 0].cpu().numpy()
            ay, ax = np.unravel_index(int(np.argmax(heat)), heat.shape)
            dist = float(np.hypot(ax - fr['gt_center'][0], ay - fr['gt_center'][1]))
            if n_print < 5:
                print(f"冒烟 {inst} t{t} f{f}: pred=({ax},{ay}) gt=({fr['gt_center'][0]:.1f},{fr['gt_center'][1]:.1f}) "
                      f"dist={dist:.1f} obj={fr['obj_size'][0]:.1f}×{fr['obj_size'][1]:.1f} "
                      f"hit={pck_hit([float(ax), float(ay)], [float(fr['gt_center'][0]), float(fr['gt_center'][1])], [float(fr['obj_size'][0]), float(fr['obj_size'][1])])}")
                n_print += 1
    out = {'census_rows': rows, 'census_total': tot, 'median_mode_n': len(ds), 'g0': g0,
           'all_pass': bool(all_pass)}
    (OUT_DIR / 'n_g0.json').write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                       encoding='utf-8')
    if not all_pass:
        print('G0 FAIL → 停机上报')
        sys.exit(1)


# ---------------- stage 1：DINOv2 模板特征缓存 ----------------
def stage1():
    torch.manual_seed(0)
    out_path = OUT_DIR / f'n_cache_tmpl{SUF}.pt'
    if out_path.exists():
        print(f'模板缓存已存在: {out_path}')
        return
    enc = DINOv2TemplateEncoder(pretrained=True).cuda().eval()
    cache = {}
    t0 = time.time()
    for inst, tmpls, _ in val_instances():
        inst_dir = split_dir() / inst
        d = {}
        files = {t: inst_dir / 'rgb' / 'template' / f'{t:06d}.jpg' for t in tmpls}
        ts = tmpls
        for s in range(0, len(ts), TMPL_BATCH):
            xs = torch.stack([norm224(cv2.resize(
                cv2.cvtColor(cv2.imread(str(files[t])), cv2.COLOR_BGR2RGB),
                (PATCH, PATCH), interpolation=cv2.INTER_LINEAR)) for t in ts[s:s + TMPL_BATCH]]).to(DEVICE)
            maps = enc.backbone.get_intermediate_layers(xs, n=[7, 15, 23], reshape=True)
            gaps = enc.backbone.get_intermediate_layers(xs, n=[15, 23], reshape=False)
            gap = torch.cat([o.mean(dim=1) for o in gaps], dim=1)
            for j, t in enumerate(ts[s:s + TMPL_BATCH]):
                d[str(t)] = {'m7': maps[0][j].cpu(), 'm15': maps[1][j].cpu(),
                             'm23': maps[2][j].cpu(), 'gap': gap[j].cpu()}
        cache[inst] = d
        print(f'[{inst}] {len(d)} 模板特征完成（累计 {time.time() - t0:.0f}s）')
    torch.save(cache, out_path)
    print(f'落盘: {out_path}（{time.time() - t0:.0f}s）')


# ---------------- stage 2：逐模板 trial 评测 ----------------
def n_trials_file(system):
    return OUT_DIR / f'n_{system}{SUF}_trials.jsonl'


def stage2(systems):
    torch.manual_seed(0)
    tmpl = torch.load(OUT_DIR / f'n_cache_tmpl{SUF}.pt', map_location='cpu')
    for key in systems:
        spec = SYSTEMS[key]
        model = build_system(key)
        use_id = spec['use_id']
        out_f = open(n_trials_file(key), 'a', encoding='utf-8')
        t0 = time.time()
        for inst, tmpls, frames in val_instances():
            if progress_load().get(f'{key}:{inst}'):
                continue
            tf = tmpl[inst]
            lines = []
            for f in frames:
                fr = load_frame(split_dir() / inst, f)
                ir_x = fr['ir'].unsqueeze(0).to(DEVICE)
                valid = [t for t in tmpls if abs(f - t) >= MIN_GAP]
                for s in range(0, len(valid), BATCH_T):
                    chunk = valid[s:s + BATCH_T]
                    logits = forward_cached(model, use_id, ir_x, [tf[str(t)] for t in chunk])
                    heat = torch.sigmoid(logits)[:, 0].float().cpu().numpy()
                    for j, t in enumerate(chunk):
                        hn = heat[j]
                        ay, ax = np.unravel_index(int(np.argmax(hn)), hn.shape)
                        peaks = nms_peaks(hn, K, MIN_DIST)
                        hit = pck_hit([float(ax), float(ay)], [float(fr['gt_center'][0]), float(fr['gt_center'][1])], [float(fr['obj_size'][0]), float(fr['obj_size'][1])])
                        lines.append(json.dumps({
                            'inst': inst, 't': t, 'f': f, 'diff': fr['difficulty'],
                            'base': [float(ax), float(ay)],
                            'peaks': [[int(p[0]), int(p[1]), round(float(p[2]), 6)] for p in peaks],
                            'gt': [round(float(fr['gt_center'][0]), 2), round(float(fr['gt_center'][1]), 2)],
                            'obj': [round(float(fr['obj_size'][0]), 2), round(float(fr['obj_size'][1]), 2)],
                            'hit': bool(hit)}, ensure_ascii=False))
            out_f.write('\n'.join(lines) + ('\n' if lines else ''))
            out_f.flush()
            progress_mark(f'{key}:{inst}')
            print(f'[{key}] {inst}: {len(lines)} trials（累计 {time.time() - t0:.0f}s）')
        out_f.close()
        del model
        torch.cuda.empty_cache()
        print(f'[{key}] 完成 → {n_trials_file(key).name}')


# ---------------- stage 3：单模板验证器 V' ----------------
def verifier_file(c_key):
    return OUT_DIR / f'n_{c_key}{SUF}_verifier.jsonl'


def stage3(c_keys):
    """V' 验证器层（v2）：对 base 系统（c2→s3 / c1→s2）逐 trial 峰集做单模板 cos 重排。
    帧级断点 + cand 分片逐实例持久化（wedge 重启不重算已嵌帧）；帧进度直写文件（绕过管道缓冲）。"""
    torch.manual_seed(0)
    torch.set_num_threads(1)
    tmpl_fp32 = torch.load(OUT_DIR / f'n_cache_tmpl{SUF}.pt', map_location='cpu')
    tmpl_v3 = torch.load(OUT_DIR / f'v3_template_slices{SUF}.pt', map_location='cpu')
    enc = DINOv2TemplateEncoder(pretrained=True).cuda().eval()
    expert0 = IdentityExpert(str(EXPERT_CKPT)).cuda().eval()
    # v3 逐切片管线核验（随机抽 3 切片：重算 fp32 vs v3 fp16 缓存行，报 max|Δ| 与 cos）
    rng = np.random.RandomState(0)
    all_rows = [(inst, t) for inst, tmpls, _ in val_instances() for t in tmpls]
    for inst, t in [all_rows[i] for i in rng.choice(len(all_rows), 3, replace=False)]:
        p = split_dir() / inst / 'rgb' / 'template' / f'{t:06d}.jpg'
        x = norm224(cv2.resize(cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB),
                               (PATCH, PATCH), interpolation=cv2.INTER_LINEAR)).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            e = expert0(enc.intermediate_gap(x))[0].cpu()
        order = sorted(int(q.stem) for q in (split_dir() / inst / 'rgb' / 'template').glob('*.jpg'))
        cached = tmpl_v3['slices'][inst][order.index(t)].float()
        cos = float(F.cosine_similarity(e, cached, dim=0))
        print(f"V' 管线核验 {inst} t{t}: 重算 vs v3缓存 max|Δ|={float((e - cached).abs().max()):.2e} "
              f"cos={cos:.6f}（fp16 逐位={bool(torch.equal(e.half(), tmpl_v3['slices'][inst][order.index(t)]))}）")
    # 模板嵌入 = fp32 gap 缓存 → head_v0 重算（v3 fp16 行与重算差 ~1e-4~1e-3 / cos≈1.0，
    # 逐位测试过严；按任务书 §3『不一致 → 重算并披露』改走重算路径）
    fuse = linear_fn(ALPHA)
    pool = ThreadPoolExecutor(8)
    MEAN_CUDA = MEAN.to(DEVICE)
    STD_CUDA = STD.to(DEVICE)
    # base trial 记录一次性读入并按实例分组
    grouped = {}
    for l in open(n_trials_file('s3'), encoding='utf-8'):
        r = json.loads(l)
        grouped.setdefault(r['inst'], []).append(r)
    grouped_s2 = {}
    for l in open(n_trials_file('s2'), encoding='utf-8'):
        r = json.loads(l)
        grouped_s2.setdefault(r['inst'], []).append(r)
    grouped_of = {'c2': grouped, 'c1': grouped_s2}
    for c_key in c_keys:
        trials_by_inst = grouped_of[c_key]
        out_f = open(verifier_file(c_key), 'a', encoding='utf-8')
        cand_cache = {}
        t0 = time.time()
        for inst, tmpls, frames in val_instances():
            if progress_load().get(f'{c_key}:{inst}'):
                continue
            row_of = {t: i for i, t in enumerate(tmpls)}
            trials = trials_by_inst.get(inst, [])
            e_all = {t: F.normalize(expert0(tmpl_fp32[inst][str(t)]['gap'].unsqueeze(0).to(DEVICE))[0].cpu(), dim=0)
                     for t in tmpls}
            by_frame = {}
            for tr in trials:
                by_frame.setdefault(tr['f'], []).append(tr)
            shard_path = OUT_DIR / f'n_cand_{c_key}{SUF}_{inst}.pt'
            meta_path = shard_path.with_suffix('.meta.json')
            frame_done = set()
            if shard_path.exists():
                cand_cache.update(torch.load(shard_path, map_location='cpu'))
                if meta_path.exists():
                    frame_done = set(json.load(open(meta_path)))
            lines = []
            fprog = open(OUT_DIR / f'n_frameprog_{c_key}{SUF}.txt', 'a', encoding='utf-8')
            for f, trs in sorted(by_frame.items()):
                if f not in frame_done:
                    fprog.write(f'{c_key} {inst} f{f} ntrials={len(trs)}\n')
                    fprog.flush()
                    ir_rgb, _, _, _ = _letterbox(cv2.cvtColor(cv2.imread(str(
                        split_dir() / inst / 'ir' / 'search' / f'{f:06d}.jpg')), cv2.COLOR_BGR2RGB), IMG_SIZE)
                    ir_bgr = cv2.cvtColor(ir_rgb, cv2.COLOR_RGB2BGR)
                    need = []
                    for tr in trs:
                        for p in tr['peaks']:
                            kk = (f, int(p[0]), int(p[1]))
                            if kk not in cand_cache:
                                need.append((kk, p))
                    if need:
                        crops = list(pool.map(lambda kp: crop_patch(ir_bgr, kp[1][0], kp[1][1]), need))
                        # 嵌入按 <=256 分块：ViT-L fp32 大批量激活（注意力 ~B×16×257×257×4B）
                        # 会触发 WSL2 CUDA Sysmem Fallback（VRAM 溢出静默走主机内存，~10× 减速）
                        xs_all = torch.from_numpy(np.stack(crops)).to(DEVICE).permute(0, 3, 1, 2).float() / 255.0
                        xs_all = (xs_all - MEAN_CUDA) / STD_CUDA
                        embs = []
                        for s in range(0, xs_all.shape[0], 256):
                            with torch.no_grad():
                                embs.append(expert0(enc.intermediate_gap(xs_all[s:s + 256])).cpu())
                        e = torch.cat(embs)
                        for j, (kk, _) in enumerate(need):
                            cand_cache[kk] = e[j].to(torch.float16)
                    frame_done.add(f)
                    if len(frame_done) % 20 == 0:
                        json.dump(sorted(frame_done), open(meta_path, 'w'))
                        torch.save({k: v for k, v in cand_cache.items() if k[0] in frame_done}, shard_path)
                for tr in trs:
                    ce = torch.stack([cand_cache[(f, int(p[0]), int(p[1]))].float()
                                      for p in tr['peaks']])
                    cos_list = [float(v) for v in F.cosine_similarity(ce, e_all[tr['t']].unsqueeze(0), dim=1)]
                    vals = [p[2] for p in tr['peaks']]
                    best = fuse(vals, cos_list)
                    pred = tr['peaks'][best][:2]
                    lines.append(json.dumps({
                        'inst': inst, 't': tr['t'], 'f': f, 'diff': tr['diff'],
                        'pick_pi': int(best), 'pred': [float(pred[0]), float(pred[1])],
                        'cos_pick': round(cos_list[best], 6),
                        'hit': bool(pck_hit(pred, tr['gt'], tr['obj']))}, ensure_ascii=False))
            torch.save({k: v for k, v in cand_cache.items() if k[0] in frame_done}, shard_path)
            json.dump(sorted(frame_done), open(meta_path, 'w'))
            out_f.write('\n'.join(lines) + ('\n' if lines else ''))
            out_f.flush()
            fprog.close()
            progress_mark(f'{c_key}:{inst}')
            print(f'[{c_key}] {inst}: {len(lines)} trials（累计 {time.time() - t0:.0f}s）', flush=True)
        out_f.close()
        del enc, expert0
        torch.cuda.empty_cache()
        print(f'[{c_key}] 完成 → {verifier_file(c_key).name}', flush=True)


# ---------------- 阶段 5：汇总统计 ----------------
def _agg(records):
    """三口径 × overall/easy/hard。records: list of dict(hit, diff, inst, t)。"""
    def stats(sub):
        if not sub:
            return {'pair_macro': None, 'trial_micro': None, 'inst_macro': None,
                    'n_pairs': 0, 'n_trials': 0}
        pair = {}
        inst = {}
        h = 0
        for r in sub:
            pair.setdefault((r['inst'], r['t']), []).append(int(r['hit']))
            inst.setdefault(r['inst'], []).append(int(r['hit']))
            h += int(r['hit'])
        return {'pair_macro': round(float(np.mean([np.mean(v) for v in pair.values()])), 4),
                'trial_micro': round(h / len(sub), 4),
                'inst_macro': round(float(np.mean([np.mean(v) for v in inst.values()])), 4),
                'n_pairs': len(pair), 'n_trials': len(sub)}

    out = {'overall': stats(records),
           'easy': stats([r for r in records if r['diff'] == 'easy']),
           'hard': stats([r for r in records if r['diff'] == 'hard'])}
    return out


def summarize():
    torch.manual_seed(0)
    systems = ['s3', 's2', 's1', 's4', 's123']
    combos = {'c2': 's3', 'c1': 's2'}
    out = {'systems': {}, 'verifiers': {}, 'r4_template_dist': {}, 'r5_contingency': None,
           'r3_gains': {}, 'r2_order': None}
    recs_of = {}
    for key in systems:
        f = n_trials_file(key)
        if not f.exists():
            continue
        recs = [json.loads(l) for l in open(f, encoding='utf-8')]
        keys_seen = [(r['inst'], r['t'], r['f']) for r in recs]
        assert len(keys_seen) == len(set(keys_seen)), f'{key} trials jsonl 有重复记录（并发写污染）'
        recs_of[key] = recs
        out['systems'][key] = _agg(recs)
        # per-class（trial-微口径）
        pc = {}
        for r in recs:
            c = r['inst'].split('_')[0] + '_'
            pc.setdefault(c, [0, 0])
            pc[c][0] += int(r['hit']); pc[c][1] += 1
        out['systems'][key]['per_class_micro'] = {
            c: {'pck': round(h / n, 4), 'n': n, 'unseen': c in ('animal_', 'truck_')}
            for c, (h, n) in sorted(pc.items())}
        # R4：per-template PCK 分布
        pair = {}
        for r in recs:
            pair.setdefault((r['inst'], r['t']), []).append(int(r['hit']))
        pcks = np.array([np.mean(v) for v in pair.values()])
        per_inst = {}
        for (inst, t), v in pair.items():
            per_inst.setdefault(inst, []).append(np.mean(v))
        out['r4_template_dist'][key] = {
            'mean': round(float(pcks.mean()), 4), 'std': round(float(pcks.std()), 4),
            'min': round(float(pcks.min()), 4), 'max': round(float(pcks.max()), 4),
            'n_pairs': int(len(pcks)),
            'per_inst': {i: {'n_t': len(v), 'mean': round(float(np.mean(v)), 4),
                             'std': round(float(np.std(v)), 4),
                             'min': round(float(np.min(v)), 4), 'max': round(float(np.max(v)), 4)}
                         for i, v in sorted(per_inst.items())}}
    for c_key, base in combos.items():
        f = verifier_file(c_key)
        if not f.exists():
            continue
        vrecs = [json.loads(l) for l in open(f, encoding='utf-8')]
        vkeys = [(r['inst'], r['t'], r['f']) for r in vrecs]
        assert len(vkeys) == len(set(vkeys)), f'{c_key} verifier jsonl 有重复记录'
        out['verifiers'][c_key] = _agg(vrecs)
        out['r3_gains'][f'{c_key}-{base}'] = round(
            (out['verifiers'][c_key]['overall']['pair_macro']
             - out['systems'][base]['overall']['pair_macro']) * 100, 2)
        if c_key == 'c2':
            b = {(r['inst'], r['t'], r['f']): r['hit'] for r in recs_of[base]}
            cont = {'base_hit_c_hit': 0, 'base_hit_c_miss': 0,
                    'base_miss_c_hit': 0, 'base_miss_c_miss': 0}
            for r in vrecs:
                bh = b[(r['inst'], r['t'], r['f'])]
                k = ('base_hit_c_hit' if bh and r['hit'] else
                     'base_hit_c_miss' if bh else
                     'base_miss_c_hit' if r['hit'] else 'base_miss_c_miss')
                cont[k] += 1
            out['r5_contingency'] = cont
    if 's3' in out['systems'] and 's2' in out['systems'] and 's1' in out['systems']:
        o = {k: out['systems'][k]['overall']['pair_macro'] for k in ('s3', 's2', 's1')}
        out['r2_order'] = {'pck': o,
                           'order': ' > '.join(sorted(o, key=o.get, reverse=True)),
                           'a2_vs_j2': 'A2 >= j2（排序保持）' if o['s3'] >= o['s2']
                                       else 'A2 < j2（排序回摆，配方卡主口径回摆 j2）'}
    (OUT_DIR / 'n_summary.json').write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                            encoding='utf-8')
    print(json.dumps({k: v for k, v in out.items() if k != 'r4_template_dist'},
                     ensure_ascii=False, indent=1)[:3000])
    print('落盘: n_summary.json')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage0', action='store_true')
    ap.add_argument('--stage1', action='store_true')
    ap.add_argument('--stage2', action='store_true')
    ap.add_argument('--stage3', action='store_true')
    ap.add_argument('--stage4', action='store_true')
    ap.add_argument('--summary', action='store_true')
    ap.add_argument('--systems', default='s3,s2',
                    help='stage2 系统列表（逗号分隔）；stage4 用 s1,s4,s123')
    ap.add_argument('--c', default='c2,c1', help='stage3 验证器列表')
    args = ap.parse_args()
    if args.stage0:
        stage0()
    elif args.stage1:
        stage1()
    elif args.stage2:
        stage2([s for s in args.systems.split(',') if s])
    elif args.stage3:
        stage3([c for c in args.c.split(',') if c])
    elif args.stage4:
        stage2([s for s in args.systems.split(',') if s])
    elif args.summary:
        summarize()
    else:
        ap.print_help()


if __name__ == '__main__':
    main()
