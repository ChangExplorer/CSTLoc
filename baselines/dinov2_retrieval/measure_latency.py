#!/usr/bin/env python
# baselines/dinov2_retrieval/measure_latency.py —— 臂③ DINOv2 特征检索推理时延
#   （复用 evaluation/measure_latency.py 的模式：3 实例 × 10 帧 = 30 帧抽样；每帧 / 每 trial 两档）
#
# 口径：逐帧 = 场景图一次前向（letterbox 384×640 → 392×644 零填充 → ViT-L 特征图）
#       + 全部合格模板(|f−t|≥30) 的逐位置余弦相关 + top-32 NMS。
#       报 场景前向 / 相关+NMS / 合计 三档。fp32、eval、no_grad。
# 用法：CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python baselines/dinov2_retrieval/measure_latency.py
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('d2run', HERE / 'run.py')
d2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d2)

N_INST, N_FRAME = 3, 10
torch.manual_seed(0)
model = d2.build_model()
census = d2.census()
rows = []

for inst, tmpls, frames in census[:N_INST]:
    V, _ = d2.vec224(model, [d2.tmpl224(inst, t) for t in tmpls])   # 主口径 257-token
    ti = {t: j for j, t in enumerate(tmpls)}
    for f in frames[:N_FRAME]:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        P = d2.scene_feat(model, inst, f)
        Pflat = P.reshape(P.shape[0], -1)
        torch.cuda.synchronize()
        dt_scene = (time.perf_counter() - t0) * 1000
        valid = [t for t in tmpls if abs(f - t) >= d2.MIN_GAP]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        S = (V[[ti[t] for t in valid]] @ Pflat).reshape(len(valid), d2.GH, d2.GW).cpu().numpy()
        for j in range(len(valid)):
            d2.nms_peaks_grid(S[j].copy())
        dt_corr = (time.perf_counter() - t0) * 1000
        n = len(valid)
        rows.append({'inst': inst, 'f': f, 'n_trials': n, 'ms_scene': round(dt_scene, 2),
                     'ms_corr_nms': round(dt_corr, 2), 'ms_frame': round(dt_scene + dt_corr, 2),
                     'ms_trial': round((dt_scene + dt_corr) / max(n, 1), 3)})
        print(f'{inst} f{f}: {n} trials, scene {dt_scene:.1f} ms + corr/NMS {dt_corr:.1f} ms '
              f'→ {dt_scene + dt_corr:.1f} ms/帧, {(dt_scene + dt_corr) / max(n, 1):.2f} ms/trial',
              flush=True)

doc = {'arm': 'dinov2_retrieval', 'backbone': 'vit_large_patch14_dinov2.lvd142m (fp32, layer 23)',
       'sampling': f'{N_INST} 实例 × {N_FRAME} 帧 = {len(rows)} 帧（实例名/帧号升序）',
       'ms_per_frame_mean': round(float(np.mean([r['ms_frame'] for r in rows])), 2),
       'ms_per_frame_median': round(float(np.median([r['ms_frame'] for r in rows])), 2),
       'ms_scene_mean': round(float(np.mean([r['ms_scene'] for r in rows])), 2),
       'ms_corr_nms_mean': round(float(np.mean([r['ms_corr_nms'] for r in rows])), 2),
       'ms_per_trial_mean': round(float(np.mean([r['ms_trial'] for r in rows])), 3),
       'device': torch.cuda.get_device_name(0), 'rows': rows}
(HERE / 'latency.json').write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding='utf-8')
print(json.dumps({k: v for k, v in doc.items() if k != 'rows'}, ensure_ascii=False, indent=1))
