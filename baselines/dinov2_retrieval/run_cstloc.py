#!/usr/bin/env python
# 臂③ DINOv2 特征检索 + 滑窗（对照实验第一批 · 任务书 2.3 §3）——GPU，零训练，冻结主干。
#
# 管线（任务书逐条）：
#   ① 冻结 DINOv2 ViT-L（timm 官方权重 vit_large_patch14_dinov2.lvd142m）整图一次前向提场景特征图：
#      raw IR → letterbox 384×640（与发布线同口径）→ 零填充到 392×644（ViT patch14 要求边长整除 14；
#      零填充不重采样、内容几何不变，见「偏差」）→ ImageNet 归一化
#      → get_intermediate_layers(n=[23], reshape=True) → (1024,28,46) → 裁内容区 (1024,27,45)
#      （patch 步长 14，27×14=378 ≤ 384、45×14=630 ≤ 640）；
#   ② 模板切片（rgb/template）resize 224×224 → 同主干 get_intermediate_layers(n=[23], reshape=False)
#      → **257-token GAP**（对 CLS+256 patch 全 token 取 mean，与仓库 intermediate_gap 同 token 口径）
#      → (1024,) L2 归一化；
#   ③ 模板-场景逐位置余弦相关（矩阵乘实现，等价 1×1 卷积）→ (27,45) 得分图 → argmax 为预测；
#   ④ top-32 峰落 peaks.jsonl（NMS 32px，原图坐标；patch 步长 42 px > 32 px → 相邻 patch 互不抑制，
#      即 top-32 = 前 32 名 patch，见报告粒度说明）。
#
# 层选择（任务书未指定）：**最后一个 block（n=[23]）**，事前固定，不做层扫描（避免事后择优）。
# 坐标系：patch (r,c) 中心 letterbox = (14c+7, 14r+7) → 原图 (x*3, (y−12)*3)。
#
# G0（同模态同源对照，分离「模态鸿沟」与「检索范式」两个因子）：
#   A 帧内自源：模板 = 同帧 IR 图在 GT 处的 224 裁窗（同模态同源，逐帧 1 模板）→ 逐帧 PCK；
#   B 协议配平：模板 = 模板帧 t 的 IR 图在 GT 处的 224 裁窗（同模态），在与主管线完全相同的
#     (inst,t,f) trial 集上评（同帧/跨帧同模态，唯一变化 = 模板来源模态）；
#   C 跨模态：模板 = RGB 模板切片（主管线）——B ≫ C 才说明候选生成确实断在模态鸿沟上。
#
# 缓存/断点续跑（新前缀 b23_*，零覆写）：
#   b23_dinov2_frame_{inst}_{首帧:06d}.pt  {'frames':[...], 'feats': (F,1024,27,45) fp32}
#   b23_dinov2_{tag}_trials.jsonl          逐 trial 底稿（唯一真源，合并按 (inst,t,f) 去重）
#   b23_dinov2_{tag}_progress.json         已完成实例
#
# 纪律：eval + no_grad；torch.manual_seed(0)；fp32（不开 TF32）；test 只跑一次；
#       产物只存原始预测（GT 由 evaluation/aggregate.py 挂接）。
#
# 用法：
#   python baselines/dinov2_retrieval/run.py --tag test          # 主线（test 全量，含 G0）
# 【执行状态 2026-10-16】本臂 test 运行未执行：臂② CLIP 重排 0.4632 命中任务书 §5 判读红线
# （≥0.4680−1pp=0.4580）→ 按令停机、不自行继续。本文件为本批任务书执行期内预先就绪的代码，
# 运行时须先取得 leader/用户明示授权。详见 docs/reports/实验结果23.md §5/§9。
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import cv2                                                                     # noqa: E402
import numpy as np                                                             # noqa: E402
import timm                                                                    # noqa: E402
import torch                                                                   # noqa: E402
import torch.nn.functional as F                                                # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / 'code'))

from data.cstloc_dataset import _letterbox, NORM_MEAN, NORM_STD                # noqa: E402
from tools.v_verifier_rerank import ROOT, IMG_SIZE, PATCH, crop_patch          # noqa: E402
ROOT = Path(os.environ.get('CSTLOC_DATA_ROOT', str(ROOT)))
from tools.v3_verifier_deepen import pck_hit                                   # noqa: E402
from tools.wrong_template_probe import load_frame                              # noqa: E402

DEVICE = 'cuda'
LAYER = 23                 # 最后一个 block（timm 0-based 索引）
PAD_HW = (392, 644)        # 14 的倍数（28×14, 46×14）；内容仍在左上 384×640
ROW0 = 1                   # 起始 patch 行：row 0 中心 y_lb=7 落在 letterbox 上黑边(pad=12)内，
                           #   该位置预测必然越界（orig y=−15），事前按几何剔除；行 1..26 中心
                           #   y_lb=21..371 全在内容区（内容 y_lb∈[12,372)）→ 26×45=1170 位置
GH, GW = 26, 45            # 特征图有效尺寸
GRID_PX = 14               # patch 步长（letterbox 像素）
NMS_DIST = 32              # 原图像素
TOPK = 32
MIN_GAP = 30
LB_SCALE, LB_PAD_Y = 3.0, 12.0
FRAME_CACHE_CHUNK = 60     # 帧缓存分片（控内存）
MEAN = torch.tensor(NORM_MEAN).view(1, 3, 1, 1)
STD = torch.tensor(NORM_STD).view(1, 3, 1, 1)
_SUP = None


def build_model():
    m = timm.create_model('vit_large_patch14_dinov2.lvd142m', pretrained=True,
                          img_size=224, dynamic_img_size=True).to(DEVICE).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


@torch.no_grad()
def scene_feat(model, inst, f):
    """整图一次前向 → (1024,27,45) 逐位置 L2 归一化特征图（letterbox 384×640 同发布线口径）。"""
    p = ROOT / 'test' / inst / 'ir' / 'search' / f'{f:06d}.jpg'
    rgb, _, _, _ = _letterbox(cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB), IMG_SIZE)
    big = np.zeros((PAD_HW[0], PAD_HW[1], 3), dtype=rgb.dtype)
    big[:IMG_SIZE[0], :IMG_SIZE[1]] = rgb                              # 左上对齐（letterbox 语义）
    # 注意：MEAN/STD 形状 (1,3,1,1)，必须先 unsqueeze 到 (1,3,H,W) 再归一化（否则广播出 5 维）
    x = torch.from_numpy(big).to(DEVICE).permute(2, 0, 1).float().div_(255.0).unsqueeze(0)
    x = (x - MEAN.to(DEVICE)) / STD.to(DEVICE)
    m = model.get_intermediate_layers(x, n=[LAYER], reshape=True)[0][0]  # (1024,28,46)：含 batch 维，取 [0]
    return F.normalize(m[:, ROW0:ROW0 + GH, :GW].contiguous(), dim=0)   # (1024,26,45)


@torch.no_grad()
def vec224(model, arrs):
    """arrs: list of (224,224,3) uint8 RGB → (主, 次) 两套 L2 归一化模板向量，各 (N,1024)。

    主口径（任务书字面「257-token GAP」）= 对 224 输入的全部 257 个 token（1 CLS + 256 patch）取 mean。
    **实测修正**：本机 timm 1.0.26 的 `get_intermediate_layers(reshape=False)` 只返回 256 个 patch token
    （不含 CLS），故必须走 `return_prefix_tokens=True` 显式取回 CLS 再拼接，才能得到字面的 257-token 均值。
    次口径（敏感度）= 256 个 patch token 均值（= 仓库 `intermediate_gap` 在当前 timm 下的实际行为）。
    """
    main, alt = [], []
    for s in range(0, len(arrs), 256):
        x = torch.from_numpy(np.stack(arrs[s:s + 256])).to(DEVICE).permute(0, 3, 1, 2).float()
        x = (x.div_(255.0) - MEAN.to(DEVICE)) / STD.to(DEVICE)
        pat, pre = model.get_intermediate_layers(x, n=[LAYER], reshape=False,
                                                 return_prefix_tokens=True)[0]   # (n,256,1024), (n,1,1024)
        main.append(F.normalize(torch.cat([pre, pat], dim=1).mean(dim=1), dim=1))   # 257-token
        alt.append(F.normalize(pat.mean(dim=1), dim=1))                             # 256-token（patch-only）
    return torch.cat(main), torch.cat(alt)


def tmpl224(inst, t):
    p = ROOT / 'test' / inst / 'rgb' / 'template' / f'{t:06d}.jpg'
    return cv2.resize(cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB),
                      (PATCH, PATCH), interpolation=cv2.INTER_LINEAR)


def letters_bgr(inst, f):
    p = ROOT / 'test' / inst / 'ir' / 'search' / f'{f:06d}.jpg'
    rgb, _, _, _ = _letterbox(cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB), IMG_SIZE)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def grid_to_orig(r, c):
    return (float((GRID_PX * c + GRID_PX // 2) * LB_SCALE),
            float((GRID_PX * (r + ROW0) + GRID_PX // 2 - LB_PAD_Y) * LB_SCALE))


def suppress_offsets(dist=NMS_DIST):
    """NMS 圆盘在 patch 网格上的偏移集（原图坐标距离 < dist，含 (0,0)）。
    本配置步长 42 原图像素 > dist=32 → 只有 (0,0)（相邻 patch 互不抑制），由几何自动导出。"""
    offs = []
    for dr in range(-GH, GH + 1):
        for dc in range(-GW, GW + 1):
            if (GRID_PX * dc * LB_SCALE) ** 2 + (GRID_PX * dr * LB_SCALE) ** 2 < dist ** 2:
                offs.append((dr, dc))
    return offs


def nms_peaks_grid(score_np, topk=TOPK, dist=NMS_DIST):
    """(27,45) 得分图上的 NMS：迭代取全局最大 + 抑制圆盘内 patch（与参照实现同几何）。"""
    global _SUP
    if _SUP is None:
        _SUP = suppress_offsets(dist)
    h = score_np
    peaks = []
    for _ in range(topk):
        idx = int(np.argmax(h))
        r, c = divmod(idx, GW)
        v = float(h[r, c])
        if not np.isfinite(v):
            break
        x, y = grid_to_orig(r, c)
        peaks.append((x, y, v))
        for dr, dc in _SUP:
            rr, cc = r + dr, c + dc
            if 0 <= rr < GH and 0 <= cc < GW:
                h[rr, cc] = -np.inf
    return peaks


def census():
    sd = ROOT / 'test'
    return [(inst, sorted(int(p.stem) for p in (sd / inst / 'rgb' / 'template').glob('*.jpg')),
             sorted(int(p.stem) for p in (sd / inst / 'ir' / 'search').glob('*.json')))
            for inst in sorted(p.name for p in sd.iterdir() if p.is_dir())]


def run(args):
    torch.manual_seed(0)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tag = args.tag
    model = build_model()
    print(f'[model] vit_large_patch14_dinov2.lvd142m layer={LAYER} dynamic_img_size=True '
          f'grid={GH}x{GW} nms_suppress_offsets={suppress_offsets()}', flush=True)
    rows = census()
    if args.instances:
        keep = set(args.instances.split(','))
        rows = [r for r in rows if r[0] in keep]
    prog = out / f'b23_dinov2_{tag}_progress.json'
    done = json.loads(prog.read_text(encoding='utf-8')) if prog.exists() else {}
    trials_path = out / f'b23_dinov2_{tag}_trials.jsonl'
    if args.smoke:
        rows = [(i, t, fr[:args.smoke]) for i, t, fr in rows]
        print(f'[smoke] 每实例只跑前 {args.smoke} 帧（数字不入报告）', flush=True)

    tot = {'n_trials': 0, 'n_frames': 0}
    g0_rows, desc_cos, t0 = [], {}, time.time()
    for inst, tmpls, frames in rows:
        if done.get(inst):
            print(f'[{inst}] 已完成，跳过', flush=True)
            continue
        with torch.no_grad():
            V, V_alt = vec224(model, [tmpl224(inst, t) for t in tmpls])     # (T,1024) 主/次
            meta, arrs_ir = {}, []
            for f in frames:
                fr = load_frame(ROOT / 'test' / inst, f)
                meta[f] = {'gt': [float(fr['gt_center'][0]), float(fr['gt_center'][1])],
                           'obj': [float(fr['obj_size'][0]), float(fr['obj_size'][1])]}
                arrs_ir.append(crop_patch(letters_bgr(inst, f), meta[f]['gt'][0], meta[f]['gt'][1]))
            # G0 同模态模板向量 = 该帧 IR 图在 GT 处的 224 裁窗（同模态、同源）→ 逐帧 1 条
            V_ir, _ = vec224(model, arrs_ir)                                # (F,1024)
        # 机制诊断：模板描述子（257-token GAP）的实例内共线性（两两 cos 均值，排除对角）
        with torch.no_grad():
            G = (V @ V.t()).cpu().numpy()
            m = ~np.eye(len(tmpls), dtype=bool)
            desc_cos[inst] = float(G[m].mean())
        ti = {t: j for j, t in enumerate(tmpls)}
        fi = {f: j for j, f in enumerate(frames)}
        lines, cache_f, cache_X, n_inst = [], [], [], 0
        for f in frames:
            P = scene_feat(model, inst, f)                                  # (1024,27,45)
            Pflat = P.reshape(P.shape[0], -1)
            cache_f.append(f)
            cache_X.append(P.cpu())          # fp32（缓存只作留存/复用，不作为计算来源）
            valid = [t for t in tmpls if abs(f - t) >= MIN_GAP]
            if valid:
                S = (V[[ti[t] for t in valid]] @ Pflat).reshape(len(valid), GH, GW).cpu().numpy()
                S_alt = (V_alt[[ti[t] for t in valid]] @ Pflat).reshape(
                    len(valid), GH, GW).cpu().numpy()
                for j, t in enumerate(valid):
                    pk = nms_peaks_grid(S[j].copy())
                    # 次口径（256-token）仅取 argmax（敏感度行用），见 README §偏离 D3-7
                    ra, ca = divmod(int(np.argmax(S_alt[j])), GW)
                    lines.append(json.dumps({
                        'inst': inst, 't': int(t), 'f': int(f), 'n_peaks': len(pk),
                        'pred': [pk[0][0], pk[0][1]],
                        'pred_pool256': [grid_to_orig(ra, ca)[0], grid_to_orig(ra, ca)[1]],
                        'peaks': [[round(p[0], 1), round(p[1], 1), round(p[2], 6)] for p in pk]},
                        ensure_ascii=False, separators=(',', ':')))
                    tot['n_trials'] += 1
            # G0：帧内自源（A）——模板 = 同一帧 IR 图在 GT 处的 224 裁窗（同模态同源），逐帧 1 模板。
            # A′（协议配平）= 把同一模板用于该帧全部合格 trial → 逐帧命中率按 trial 数加权
            #（模板向量与 trial 的 t 无关，故 A′ = A 的 trial 加权平均，无需重复计算）。
            gt, obj = meta[f]['gt'], meta[f]['obj']
            pk_a = nms_peaks_grid(((V_ir[fi[f]].reshape(1, -1)) @ Pflat)
                                  .reshape(GH, GW).cpu().numpy().copy())
            A_hit = float(pck_hit(list(pk_a[0][:2]), gt, obj))
            g0_rows.append({'inst': inst, 'f': f, 'A_hit': A_hit, 'n_valid': len(valid),
                            'thr_px': 0.5 * max(obj)})
            tot['n_frames'] += 1
            n_inst += 1
            if n_inst % FRAME_CACHE_CHUNK == 0:
                torch.save({'frames': cache_f, 'feats': torch.stack(cache_X)},
                           out / f'b23_dinov2_frame_{inst}_{cache_f[0]:06d}.pt')
                cache_f, cache_X = [], []
            if tot['n_frames'] % 20 == 0:
                print(f'    {inst} f{f}: 累计 trial {tot["n_trials"]}（{time.time() - t0:.0f}s）',
                      flush=True)
        if cache_f:
            torch.save({'frames': cache_f, 'feats': torch.stack(cache_X)},
                       out / f'b23_dinov2_frame_{inst}_{cache_f[0]:06d}.pt')
        with open(trials_path, 'a', encoding='utf-8') as fh:
            fh.write('\n'.join(lines) + ('\n' if lines else ''))
        n_new = len(lines)
        with open(out / f'b23_dinov2_{tag}_g0_rows.json', 'w', encoding='utf-8') as fh:
            json.dump(g0_rows, fh)
        done[inst] = 1
        prog.write_text(json.dumps(done, ensure_ascii=False, indent=1), encoding='utf-8')
        print(f'[{inst}] 新 trial={n_new} 帧={len(frames)} 完成（累计 {tot["n_trials"]}，'
              f'{time.time() - t0:.0f}s）', flush=True)
        del V, V_ir
    print(f'[done] trials={tot["n_trials"]} 用时 {time.time() - t0:.0f}s', flush=True)

    (out / f'b23_dinov2_{tag}_desc_collinearity.json').write_text(
        json.dumps({'metric': '实例内模板描述子两两余弦均值（257-token GAP，排除对角）',
                    'per_inst': desc_cos,
                    'overall_mean': round(float(np.mean(list(desc_cos.values()))), 4)
                    if desc_cos else None}, ensure_ascii=False, indent=1), encoding='utf-8')
    if not args.smoke and Path(out / f'b23_dinov2_{tag}_g0_rows.json').exists():
        g0_rows = json.load(open(out / f'b23_dinov2_{tag}_g0_rows.json', encoding='utf-8'))
        A = float(np.mean([r['A_hit'] for r in g0_rows]))
        nv = sum(r['n_valid'] for r in g0_rows)
        A_ = float(sum(r['A_hit'] * r['n_valid'] for r in g0_rows) / nv) if nv else None
        doc = {'gate': 'G0 同模态同源对照（分离「模态鸿沟」与「检索范式」两因子）',
               'prescribed': '任务书 §3：同一帧的 IR 图裁剪当模板（同模态同源）→ 检索命中率应 ≫ 跨模态命中率',
               'n_frames': len(g0_rows), 'n_trials_covered': nv,
               'A_same_frame_source_pck_frame_weighted': round(A, 4),
               'A2_same_modality_protocol_matched_pck_trial_weighted': round(A_, 4) if A_ is not None else None,
               'C_cross_modal_pck': '由 evaluation/aggregate.py 给（主管线口径，同 307,527 trial 集）',
               'note': ('模板 = 同帧 IR 图在 GT 处的 224 裁窗（与候选裁窗同一条 crop_patch/224 管线）；'
                        'A′ = 同一模板向量用于该帧全部合格 trial，故为 A 的 trial 加权平均；'
                        'A′ 与 C 同 trial 集、同相关式、同 argmax/NMS 口径，可直接比。'),
               'lambda_ceiling_px': 42, 'rows': g0_rows}
        (out / f'b23_dinov2_{tag}_g0.json').write_text(
            json.dumps(doc, ensure_ascii=False, indent=1), encoding='utf-8')
        print(f'[G0] A(同模态同源，逐帧)={A:.4f}  A′(同模态协议配平)={A_:.4f}', flush=True)

    merge(out, tag)


def merge(out, tag):
    """主口径（257-token GAP）→ 契约产物 results/peaks；次口径（256-token）→ 派生 results（敏感度行）。"""
    res = out / f'b23_dinov2_{tag}_results.jsonl'
    pl = out / f'b23_dinov2_{tag}_peaks.jsonl'
    res2 = out / f'b23_dinov2_{tag}_results_pool256.jsonl'
    seen, n = set(), 0
    with open(out / f'b23_dinov2_{tag}_trials.jsonl', encoding='utf-8') as fh, \
         open(res, 'w', encoding='utf-8') as rf, open(pl, 'w', encoding='utf-8') as pf, \
         open(res2, 'w', encoding='utf-8') as r2:
        for ln in fh:
            r = json.loads(ln)
            k = (r['inst'], r['t'], r['f'])
            if k in seen:
                continue
            seen.add(k)
            rf.write(json.dumps({'inst': r['inst'], 'tmpl_mod': 'rgb', 't': r['t'],
                                 'scene_mod': 'ir', 'f': r['f'], 'pred': r['pred']},
                                separators=(',', ':')) + '\n')
            pf.write(json.dumps({'inst': r['inst'], 't': r['t'], 'f': r['f'],
                                 'peaks': r['peaks']}, separators=(',', ':')) + '\n')
            if 'pred_pool256' in r:
                r2.write(json.dumps({'inst': r['inst'], 'tmpl_mod': 'rgb', 't': r['t'],
                                     'scene_mod': 'ir', 'f': r['f'],
                                     'pred': r['pred_pool256']}, separators=(',', ':')) + '\n')
            n += 1
    print(f'[merge] {n} trials → {res.name} / {pl.name}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='test')
    ap.add_argument('--out-dir', default=str(HERE))
    ap.add_argument('--instances', default=None)
    ap.add_argument('--smoke', type=int, default=None, help='每实例只跑前 N 帧（冒烟）')
    a = ap.parse_args()
    run(a)


if __name__ == '__main__':
    main()
