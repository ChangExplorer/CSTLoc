# baselines/ncc/ —— 臂① NCC 模板匹配（纯 CPU）

任务书 2.3 §1。免费下限对照：灰度 + `cv2.matchTemplate(TM_CCOEFF_NORMED)`，模板三尺度
{0.75×, 1×, 1.5×} 全图扫描，取全图全尺度响应最大点为预测，并落 top-32 响应峰（NMS 32px）。

## 环境

- 解释器：`/home/user/anaconda3/envs/viir/bin/python`（仅用 cv2 / numpy，**不占 GPU**）
- 数据集（只读）：`/home/user/Code/Datasets/CSTLoc-UAV_v2/test`
- 并发：`multiprocessing.Pool`，每 worker `cv2.setNumThreads(1)`（本 build 的 `matchTemplate`
  实测单线程，并行靠多进程）

## 命令

```bash
# 全量 test（307,527 trials；分片可断点续跑，重跑自动跳过已完成分片）
python baselines/ncc/run_cstloc.py --split test --workers 12 --chunk 8 --tag test \
    --check-trialset code/diag_out/e130_s2024_test_peaks32.jsonl

# G0 同源自检（场景帧自身裁剪当模板，峰值须落回裁剪中心，容差 2px）
python baselines/ncc/run_cstloc.py --g0 --split test --g0-out baselines/ncc/g0_ncc_selfcheck.json

# 冒烟（≤200 trials，数字不入报告；两次运行须逐位一致）
python baselines/ncc/run_cstloc.py --split test --smoke 200 --out-dir /tmp/ncc_smoke --tag smoke

# 时延（30 帧抽样；须 CPU 空闲）
python baselines/ncc/measure_latency.py
```

## 产物

| 文件 | 内容 |
|---|---|
| `results.jsonl` | `{"inst","tmpl_mod","t","scene_mod","f","pred":[x,y]}`，**原图 1920×1080 像素** |
| `peaks.jsonl` | `{"inst","t","f","peaks":[[x,y,score]...]}`，top-32（NMS 32px），降序 |
| `summary.json` | 清单/尺度跳过计数/耗时/分片统计 |
| `g0_ncc_selfcheck.json` | G0 同源自检逐例证据 |
| `trialset_check.json` | 与发布线峰值底稿的 trial 键集比对 |
| `shards/*.res.jsonl` `shards/*.peak.jsonl` | 分片底稿（断点续跑单元，= results/peaks 的拼接源） |
| `latency.json` | 30 帧时延（ms/帧、ms/trial） |

## 实现细则（与任务书写法的对应与偏离）

1. **尺度合并**：`TM_CCOEFF_NORMED` 响应坐标是模板窗口**左上角**，三尺度窗口尺寸不同，直接逐点
   比较会引入尺度相关偏移。故每个尺度响应按 `+(th//2, tw//2)` 平移到**窗口中心**坐标系后取逐点
   max，得到「同一位置同一含义」的全尺度响应图 S（H×W 原图像素），argmax 与 NMS 均在 S 上做。
   `pred` 即该 trial 的 S 全局 argmax（= `peaks[0]`）。
2. **尺度可行性**：缩放后模板任一维超过图像维（cv2 会直接报错）则该尺度跳过并计数
   （`summary.json: n_scale_skipped`）。实测 test 全量三尺度均可行（见 summary）。
3. **NMS 口径**：几何与 `code/tools/v_verifier_rerank.nms_peaks` 一致（迭代全局最大 + 抑制
   半径 < 32px 圆盘）。**唯一刻意偏离**：参照实现在 heat（sigmoid∈(0,1)）口径下遇 `v≤0` 即截断，
   NCC 响应可为负，故不沿用该截断，始终取满 top-32（不足如实少记）；未覆盖位（-1e30）不入峰表。
4. **插值**：下采样 `INTER_AREA`、上采样 `INTER_LINEAR`（与仓库 `_letterbox` 同族）。
5. **确定性**：无随机、无梯度；分片固定 → 合并顺序固定（实例名升序 / 帧号升序 / 模板号升序）。
   同配置双跑逐位一致（冒烟两次 sha256 见证）。
6. 坐标为原图原生 1920×1080（本臂全程无 letterbox，不涉及坐标逆变换）。
