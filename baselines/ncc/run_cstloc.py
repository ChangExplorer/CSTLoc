#!/usr/bin/env python
# 臂① NCC 模板匹配（对照实验第一批 · 任务书 2.3 §1）——纯 CPU，零训练，零 GPU。
#
# 设计（任务书逐字）：
#   · 模板（可见光切片 rgb/template/{t:06d}.jpg）与 IR 场景图都转灰度；
#   · cv2.matchTemplate(TM_CCOEFF_NORMED)，模板按 {0.75×, 1×, 1.5×} 三尺度各扫一遍；
#   · 取「全图全尺度」响应最大点为预测（pred = 该 trial 的 argmax）；
#   · 同时落 top-32 响应峰（NMS 32px）为 peaks.jsonl。
#
# 口径与实现细则（全部落进报告「偏差」节）：
#   1) 尺度响应对齐：TM_CCOEFF_NORMED 的响应坐标是**模板窗口左上角**；三尺度窗口尺寸不同，
#      直接把响应图逐点比较会让大尺度系统性偏移。故把每个尺度响应按 `+ (th//2, tw//2)` 平移到
#      **窗口中心**坐标系后取逐点 max，得到「同一位置同一含义」的全尺度响应图 S（H×W，原图像素）。
#      未覆盖位置填 -1e30（不参与 argmax）；argmax 与 NMS 均在 S 上进行。
#   2) 尺度不可行（缩放后模板任一维 > 图像该维）时该尺度跳过并计数（cv2 会直接报错）。
#      插值：下采样 INTER_AREA，上采样 INTER_LINEAR（与仓库 _letterbox 同族）。
#   3) 参照实现 tools/v_verifier_rerank.nms_peaks 在 heat（sigmoid∈(0,1)）口径下遇 v<=0 即截断；
#      NCC 响应可为负，故**不沿用该截断**，始终取满 top-32（不足如实少记）——唯一刻意偏离，
#      已在报告明卷。非有限值（未覆盖位）不入峰表。
#   4) 坐标为场景图原始 1920×1080 像素（NCC 全程原生尺度，无 letterbox 转换）。
#   5) 确定性：cv2.setNumThreads(1)（实测本 build 的 matchTemplate 单线程）+ 固定分片；
#      eval 语义天然满足（无随机、无梯度）。同配置双跑逐位一致由 --smoke 两次 sha256 见证。
#
# trial 集 = {(inst, t, f) : |f−t| ≥ 30}，与发布线峰值底稿 e130_s2024_test_peaks32.jsonl 同集
# （--check-trialset 逐键比对）。
#
# 用法：
#   # 冒烟（≤200 trials，数字只用于管线自检，不入报告）
#   python baselines/ncc/run.py --split test --smoke 200 --out-dir /tmp/ncc_smoke
#   # 全量（test 307,527 trials）
#   python baselines/ncc/run.py --split test
#   # G0 同源自检
#   python baselines/ncc/run.py --g0
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

cv2.setNumThreads(1)

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ROOT_DEFAULT = os.environ.get('CSTLOC_DATA_ROOT', '/home/user/Code/Datasets/CSTLoc')
MIN_GAP = 30
SCALES = (0.75, 1.0, 1.5)
NMS_DIST = 32
TOPK = 32
NEG = -1e30

# NMS 圆盘（严格 < r²，与 tools/v_verifier_rerank.nms_peaks 口径一致）
_OFF = np.arange(-(NMS_DIST - 1), NMS_DIST)
_dy, _dx = np.meshgrid(_OFF, _OFF, indexing='ij')
DISK = (_dx * _dx + _dy * _dy) < NMS_DIST ** 2


def nms_peaks(resp, k=TOPK, dist=NMS_DIST):
    """迭代取全局最大 + 抑制半径 dist 圆盘，返回 [(x, y, val)] 降序。与参照实现同 NMS 几何。"""
    h = resp.copy()
    H, W = h.shape
    peaks = []
    for _ in range(k):
        idx = int(np.argmax(h))
        y, x = divmod(idx, W)
        v = float(h[y, x])
        if not np.isfinite(v) or v <= NEG / 2:
            break
        peaks.append((x, y, v))
        y0, y1 = y - (dist - 1), y + dist
        x0, x1 = x - (dist - 1), x + dist
        cy0, cy1 = max(0, y0), min(H, y1)
        cx0, cx1 = max(0, x0), min(W, x1)
        if cy1 <= cy0 or cx1 <= cx0:
            continue
        h[cy0:cy1, cx0:cx1][DISK[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0]] = 0.0
    return peaks


def scale_template(g, s):
    if s == 1.0:
        return g
    interp = cv2.INTER_AREA if s < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(g, None, fx=s, fy=s, interpolation=interp)


def trial_resp(gray, tmpl_gray, scales):
    """三尺度响应 → 中心对齐的全图 max 响应图 S（H×W）+ 逐尺度可行性记录。"""
    H, W = gray.shape[:2]
    S = np.full((H, W), NEG, dtype=np.float32)
    n_used = n_skip = 0
    for s in scales:
        ts = scale_template(tmpl_gray, s)
        th, tw = ts.shape[:2]
        if th > H or tw > W:
            n_skip += 1
            continue
        r = cv2.matchTemplate(gray, ts, cv2.TM_CCOEFF_NORMED)
        oy, ox = th // 2, tw // 2
        np.maximum(S[oy:oy + r.shape[0], ox:ox + r.shape[1]], r,
                   out=S[oy:oy + r.shape[0], ox:ox + r.shape[1]])
        n_used += 1
    return S, n_used, n_skip


# ---------------- 清单 ----------------
def census(root, split, instances=None, limit=None):
    sd = Path(root) / split
    rows = []
    for inst in sorted(p.name for p in sd.iterdir() if p.is_dir()):
        if instances and inst not in instances:
            continue
        tmpls = sorted(int(p.stem) for p in (sd / inst / 'rgb' / 'template').glob('*.jpg'))
        frames = sorted(int(p.stem) for p in (sd / inst / 'ir' / 'search').glob('*.json'))
        rows.append((inst, tmpls, frames))
    tot = {'instances': len(rows), 'templates': sum(len(t) for _, t, _ in rows),
           'frames': sum(len(f) for _, _, f in rows),
           'trials': sum(sum(1 for f in fr for t in tm if abs(f - t) >= MIN_GAP)
                         for _, tm, fr in rows)}
    return rows, tot


# ---------------- worker ----------------
_G = {}


def _init(root, split):
    cv2.setNumThreads(1)
    _G['root'] = root
    _G['split'] = split


def _work(task):
    """task = (inst, [frames], [tmpls], scales, shard_path)。写分片，返回统计。"""
    inst, frames, tmpls, scales, shard_path = task
    root, split = _G['root'], _G['split']
    tdir = Path(root) / split / inst / 'rgb' / 'template'
    sdir = Path(root) / split / inst / 'ir' / 'search'
    # 该实例全部模板灰度 + 三尺度缓存（内存内；1666 张全 test 级模板体积可控）
    tg = {t: cv2.imread(str(tdir / f'{t:06d}.jpg'), cv2.IMREAD_GRAYSCALE) for t in tmpls}
    miss = [t for t, g in tg.items() if g is None]
    assert not miss, f'{inst}: 模板解码失败 {miss[:3]}'
    tmp_scale = {(t, s): scale_template(tg[t], s) for t in tmpls for s in scales}
    t_avail = {t: [s for s in scales
                   if tmp_scale[(t, s)].shape[0] <= 1080 and tmp_scale[(t, s)].shape[1] <= 1920]
               for t in tmpls}
    st = {'inst': inst, 'n_frames': len(frames), 'n_trials': 0, 'n_scale_used': 0,
          'n_scale_skipped': 0, 'n_trials_missing_scale': 0}
    rl, pl = [], []
    for f in frames:
        fpath = sdir / f'{f:06d}.jpg'
        assert fpath.exists(), f'{inst}/{f}: 缺 IR 搜索帧 jpg'
        gray = cv2.imread(str(fpath), cv2.IMREAD_GRAYSCALE)
        assert gray is not None and gray.shape == (1080, 1920), f'{inst}/{f}: 帧异常 {None if gray is None else gray.shape}'
        H, W = gray.shape[:2]
        for t in tmpls:
            if abs(f - t) < MIN_GAP:
                continue
            S = np.full((H, W), NEG, dtype=np.float32)
            used = skip = 0
            for s in t_avail[t]:
                ts = tmp_scale[(t, s)]
                th, tw = ts.shape[:2]
                r = cv2.matchTemplate(gray, ts, cv2.TM_CCOEFF_NORMED)
                oy, ox = th // 2, tw // 2
                np.maximum(S[oy:oy + r.shape[0], ox:ox + r.shape[1]], r,
                           out=S[oy:oy + r.shape[0], ox:ox + r.shape[1]])
                used += 1
            skip = len(scales) - used
            if used == 0:
                continue
            peaks = nms_peaks(S)
            st['n_trials'] += 1
            st['n_scale_used'] += used
            st['n_scale_skipped'] += skip
            if skip:
                st['n_trials_missing_scale'] += 1
            ax, ay = int(peaks[0][0]), int(peaks[0][1])
            rl.append(json.dumps({'inst': inst, 'tmpl_mod': 'rgb', 't': int(t),
                                  'scene_mod': 'ir', 'f': int(f), 'pred': [ax, ay]},
                                 separators=(',', ':')))
            pl.append(json.dumps({'inst': inst, 't': int(t), 'f': int(f),
                                  'peaks': [[int(p[0]), int(p[1]), round(float(p[2]), 6)]
                                            for p in peaks if np.isfinite(p[2])]},
                                 separators=(',', ':')))
    shard_path = Path(shard_path)
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    shard_path.with_suffix('.res.jsonl').write_text('\n'.join(rl) + ('\n' if rl else ''),
                                                   encoding='utf-8')
    shard_path.with_suffix('.peak.jsonl').write_text('\n'.join(pl) + ('\n' if pl else ''),
                                                     encoding='utf-8')
    st['status'] = 'ok'
    return st


# ---------------- G0 同源自检 ----------------
def run_g0(root, split, out_path, instances=('animal_004', 'bus_025', 'car_008', 'pedestrian_194'),
           sizes=(64, 96, 160, 224), frames_per_inst=2, tol=2.0):
    """同源自检：用场景帧（IR）自身裁剪当模板，在同一帧上匹配，峰值须落回裁剪中心（容差 2px）。
    两档口径：① 仅 1× 尺度（响应几何/坐标映射的最严格校验）；② 三尺度全管线（尺度合并
    中心对齐的校验——1× 处 NCC=1.0 仍是全尺度全局最大，若中心对齐写错则 argmax 会偏 th/4）。"""
    rows = []
    for inst in instances:
        sdir = Path(root) / split / inst / 'ir' / 'search'
        jfs = sorted(sdir.glob('*.json'))[:frames_per_inst]
        for jf in jfs:
            f = int(jf.stem)
            gray = cv2.imread(str(sdir / f'{f:06d}.jpg'), cv2.IMREAD_GRAYSCALE)
            H, W = gray.shape[:2]
            meta = json.loads(jf.read_text(encoding='utf-8'))
            tgt = [s for s in meta['shapes'] if (s.get('flags') or {}).get('target')]
            assert tgt, f'{inst}/{f}: 无 target 标注'
            pts = np.asarray(tgt[0]['points'], dtype=float)
            cx, cy = float(pts[:, 0].mean()), float(pts[:, 1].mean())
            for S in sizes:
                # 以 GT 中心为裁剪中心；越界则整体平移（记录实际中心）
                x0 = int(round(cx)) - S // 2
                y0 = int(round(cy)) - S // 2
                x0 = min(max(0, x0), W - S)
                y0 = min(max(0, y0), H - S)
                patch = gray[y0:y0 + S, x0:x0 + S].copy()
                ecx, ecy = x0 + S // 2, y0 + S // 2     # 裁剪中心（=模板中心）
                rec = {'inst': inst, 'f': f, 'win': S, 'crop_topleft': [x0, y0],
                       'expected_center': [ecx, ecy], 'gt_center': [round(cx, 2), round(cy, 2)],
                       'tol_px': tol}
                for tag, scales in (('1x', (1.0,)), ('3scale', SCALES)):
                    Spy, used, skip = trial_resp(gray, patch, scales)
                    v_exp = float(Spy[ecy, ecx])
                    ay, ax = np.unravel_index(int(np.argmax(Spy)), Spy.shape)
                    d = float(np.hypot(ax - ecx, ay - ecy))
                    rec[tag] = {'argmax': [int(ax), int(ay)], 'dist_px': round(d, 4),
                                'resp_at_expected': round(v_exp, 6),
                                'resp_at_argmax': round(float(Spy[ay, ax]), 6),
                                'n_scale_used': used, 'n_scale_skipped': skip,
                                'pass': bool(d <= tol)}
                rec['pass'] = rec['1x']['pass'] and rec['3scale']['pass']
                rows.append(rec)
    ok = sum(r['pass'] for r in rows)
    doc = {'gate': 'G0 同源自检（场景帧自身裁剪当模板 → 峰值须落回裁剪中心，容差 2px）',
           'instances': list(instances), 'sizes': list(sizes),
           'frames_per_inst': frames_per_inst, 'n_cases': len(rows), 'n_pass': ok,
           'pass': ok == len(rows), 'cases': rows}
    if out_path:
        Path(out_path).write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'[G0] 同源自检 {ok}/{len(rows)} 通过 → {out_path}', flush=True)
    return doc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='test')
    ap.add_argument('--data-root', default=ROOT_DEFAULT)
    ap.add_argument('--out-dir', default=str(HERE))
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--chunk', type=int, default=8, help='每个任务处理多少帧')
    ap.add_argument('--instances', default=None, help='逗号分隔；缺省全集')
    ap.add_argument('--smoke', type=int, default=None, help='只跑前 N 个 trial（按 inst/f/t 序）')
    ap.add_argument('--scales', default='0.75,1.0,1.5')
    ap.add_argument('--check-trialset', default=None, help='与既有峰值底稿逐键比对')
    ap.add_argument('--tag', default=None, help='分片/日志标签（默认 split）')
    ap.add_argument('--g0', action='store_true', help='只跑 G0 同源自检')
    ap.add_argument('--g0-out', default=None)
    a = ap.parse_args()
    if a.g0:
        run_g0(a.data_root, a.split, a.g0_out)
        return
    scales = tuple(float(x) for x in a.scales.split(','))
    assert scales == SCALES, '任务书写死三尺度 {0.75,1,1.5}'
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tag = a.tag or a.split
    insts = a.instances.split(',') if a.instances else None

    rows, tot = census(a.data_root, a.split, insts)
    print(f'[census] split={a.split} instances={tot["instances"]} templates={tot["templates"]} '
          f'frames={tot["frames"]} trials={tot["trials"]}', flush=True)

    # 分片切分（inst × 连续帧块），保留确定性顺序
    tasks = []
    smoke_left = a.smoke
    for inst, tmpls, frames in rows:
        i = 0
        while i < len(frames):
            ch = frames[i:i + a.chunk]
            if smoke_left is not None:
                ntr = sum(1 for f in ch for t in tmpls if abs(f - t) >= MIN_GAP)
                if smoke_left <= 0:
                    break
                if ntr > smoke_left:      # 冒烟截断（帧粒度、不越界：截断后 trial 数 ≤ smoke）
                    keep, c = [], 0
                    for f in ch:
                        nf = sum(1 for t in tmpls if abs(f - t) >= MIN_GAP)
                        if c + nf > smoke_left:
                            break
                        keep.append(f)
                        c += nf
                    smoke_left -= c
                    if keep:
                        tasks.append((inst, keep, tmpls, scales,
                                      str(out / 'shards' / f'{tag}_{inst}_{keep[0]:06d}.jsonl')))
                    break
                smoke_left -= ntr
            tasks.append((inst, ch, tmpls, scales,
                          str(out / 'shards' / f'{tag}_{inst}_{ch[0]:06d}.jsonl')))
            i += a.chunk
        if smoke_left is not None and smoke_left <= 0:
            break
    todo = [t for t in tasks if not Path(t[4]).with_suffix('.res.jsonl').exists()]
    print(f'[shards] 总 {len(tasks)} 片，待跑 {len(todo)} 片，workers={a.workers}', flush=True)

    t0 = time.time()
    stats, done = [], 0
    with Pool(a.workers, initializer=_init, initargs=(a.data_root, a.split)) as pool:
        for st in pool.imap_unordered(_work, todo):
            stats.append(st)
            done += 1
            el = time.time() - t0
            print(f'[{done}/{len(todo)}] {st["inst"]} frames={st["n_frames"]} '
                  f'trials={st["n_trials"]} skip_scale={st["n_scale_skipped"]} '
                  f'({el:.0f}s, {st["n_trials"] / max(el, 1e-9) * len(todo) / max(done, 1):.0f} trials/s proj)',
                  flush=True)
    print(f'[run] 完成 {len(stats)} 片，用时 {time.time() - t0:.0f}s', flush=True)

    # 合并（确定性顺序：分片按 (inst, 首帧号) 升序；分片内按 inst/f/t 序）。冒烟截断的分片只含
    # 其截断后的行，后续同块分片不存在 → 直接跳过（结果集 = 已跑分片的并集）。
    nres = npk = 0
    res_path = out / f'results{"" if tag == "test" else "_" + tag}.jsonl'
    pk_path = out / f'peaks{"" if tag == "test" else "_" + tag}.jsonl'
    with open(res_path, 'w', encoding='utf-8') as res_f, open(pk_path, 'w', encoding='utf-8') as pk_f:
        for _, _, shard in sorted(((t[0], t[1][0], t[4]) for t in tasks), key=lambda x: (x[0], x[1])):
            rf, pf = Path(shard).with_suffix('.res.jsonl'), Path(shard).with_suffix('.peak.jsonl')
            if not rf.exists():
                continue
            txt = rf.read_text(encoding='utf-8')
            res_f.write(txt)
            nres += txt.count('\n')
            ptxt = pf.read_text(encoding='utf-8')
            pk_f.write(ptxt)
            npk += ptxt.count('\n')
    print(f'[merge] results={nres} peaks={npk} → {res_path.name}', flush=True)

    # 统计落盘
    summ = {'split': a.split, 'scales': list(scales), 'min_gap': MIN_GAP, 'nms_dist': NMS_DIST,
            'topk': TOPK, 'census': tot, 'n_results': nres, 'n_peaks': npk,
            'n_scale_used': sum(s['n_scale_used'] for s in stats),
            'n_scale_skipped': sum(s['n_scale_skipped'] for s in stats),
            'n_trials_missing_scale': sum(s['n_trials_missing_scale'] for s in stats),
            'elapsed_s': round(time.time() - t0, 1),
            'per_shard': sorted(stats, key=lambda s: (s['inst'], s['n_frames']))}
    (out / f'summary{"" if tag == "test" else "_" + tag}.json').write_text(
        json.dumps(summ, ensure_ascii=False, indent=1), encoding='utf-8')
    if a.check_trialset:
        keys_ref = set()
        for ln in open(a.check_trialset, encoding='utf-8'):
            r = json.loads(ln)
            keys_ref.add((r['inst'], int(r['t']), int(r['f'])))
        keys_got = set()
        for ln in open(out / f'results{"" if tag == "test" else "_" + tag}.jsonl', encoding='utf-8'):
            r = json.loads(ln)
            keys_got.add((r['inst'], int(r['t']), int(r['f'])))
        chk = {'ref': a.check_trialset, 'n_ref': len(keys_ref), 'n_got': len(keys_got),
               'only_in_ref': sorted(keys_ref - keys_got)[:5],
               'only_in_got': sorted(keys_got - keys_ref)[:5],
               'identical': keys_ref == keys_got}
        (out / 'trialset_check.json').write_text(json.dumps(chk, ensure_ascii=False, indent=1),
                                                encoding='utf-8')
        print(f"[trialset] ref={len(keys_ref)} got={len(keys_got)} identical={chk['identical']}",
              flush=True)


if __name__ == '__main__':
    main()
