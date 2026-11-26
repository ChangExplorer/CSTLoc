#!/usr/bin/env python
# baselines/ncc/measure_latency.py —— 臂① NCC 推理时延（复用 evaluation/measure_latency.py 的模式）
#
# 口径：3 实例 × 10 帧 = 30 帧抽样（与 evaluation/measure_latency.py 同抽样规则：实例名升序前 3、
#       帧号升序前 10）；单流串行（cv2 单线程），报 每帧 / 每 trial 两档（ms）。
#       每帧耗时 = 该帧全部合格模板(|f−t|≥30) × 三尺度 matchTemplate + 全尺度合并 + top-32 NMS；
#       每 trial = 每帧耗时 / 该帧合格模板数。
# 用法：python baselines/ncc/measure_latency.py            # 需 CPU 空闲（勿与批量作业并行）
import importlib.util
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('nccrun', HERE / 'run.py')
ncc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ncc)
cv2.setNumThreads(1)

N_INST, N_FRAME = 3, 10
rows, tot_f, tot_t = [], 0.0, 0.0
_census_rows, _ = ncc.census(ncc.ROOT_DEFAULT, 'test')
for inst, tmpls, frames in _census_rows[:N_INST]:
    tdir = Path(ncc.ROOT_DEFAULT) / 'test' / inst / 'rgb' / 'template'
    sdir = Path(ncc.ROOT_DEFAULT) / 'test' / inst / 'ir' / 'search'
    tg = {t: cv2.imread(str(tdir / f'{t:06d}.jpg'), cv2.IMREAD_GRAYSCALE) for t in tmpls}
    sc = {(t, s): ncc.scale_template(tg[t], s) for t in tmpls for s in ncc.SCALES}
    for f in frames[:N_FRAME]:
        gray = cv2.imread(str(sdir / f'{f:06d}.jpg'), cv2.IMREAD_GRAYSCALE)
        H, W = gray.shape[:2]
        valid = [t for t in tmpls if abs(f - t) >= ncc.MIN_GAP]
        t0 = time.perf_counter()
        for t in valid:
            S = np.full((H, W), ncc.NEG, dtype=np.float32)
            for s in ncc.SCALES:
                ts = sc[(t, s)]
                if ts.shape[0] > H or ts.shape[1] > W:
                    continue
                r = cv2.matchTemplate(gray, ts, cv2.TM_CCOEFF_NORMED)
                oy, ox = ts.shape[0] // 2, ts.shape[1] // 2
                np.maximum(S[oy:oy + r.shape[0], ox:ox + r.shape[1]], r,
                           out=S[oy:oy + r.shape[0], ox:ox + r.shape[1]])
            ncc.nms_peaks(S)
        dt = (time.perf_counter() - t0) * 1000
        rows.append({'inst': inst, 'f': f, 'n_trials': len(valid), 'ms_frame': round(dt, 2),
                     'ms_trial': round(dt / max(len(valid), 1), 3)})
        tot_f += dt
        tot_t += dt / max(len(valid), 1)
        print(f'{inst} f{f}: {len(valid)} trials, {dt:.1f} ms/帧, {dt / max(len(valid), 1):.2f} ms/trial',
              flush=True)

doc = {'arm': 'ncc', 'sampling': f'{N_INST} 实例 × {N_FRAME} 帧 = {len(rows)} 帧（实例名/帧号升序）',
       'ms_per_frame_mean': round(tot_f / len(rows), 2), 'ms_per_frame_median':
       round(float(np.median([r['ms_frame'] for r in rows])), 2),
       'ms_per_trial_mean': round(tot_t / len(rows), 3),
       'threads': 1, 'device': 'cpu', 'rows': rows}
(HERE / 'latency.json').write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding='utf-8')
print(json.dumps({k: v for k, v in doc.items() if k != 'rows'}, ensure_ascii=False, indent=1))
