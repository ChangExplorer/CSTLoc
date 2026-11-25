# NCC 对照臂（纯 CPU）

灰度化后用 cv2.matchTemplate(TM_CCOEFF_NORMED) 做模板匹配。模板按 {0.75×, 1×, 1.5×}
三尺度各扫描一遍，取全图全尺度响应最大点为预测，并落 top-32 响应峰（NMS 32px）。
该臂不使用学习，是主表的经典匹配下限。

## 环境与数据

- 解释器：/home/user/anaconda3/envs/viir/bin/python（仅 cv2 / numpy，不占 GPU）
- 数据：CSTLOC_DATA_ROOT，默认 /home/user/Code/Datasets/CSTLoc
- 并发：multiprocessing.Pool，每 worker cv2.setNumThreads(1)；并行靠多进程

## 命令

~~~bash
python baselines/ncc/run_cstloc.py --split test --workers 12 --chunk 8 --tag test
~~~

分片可断点续跑，重跑自动跳过已完成分片。

## v3 test 结果

| 指标 | NCC |
|---|---:|
| pair-macro PCK@0.5·max | 0.0633 |
| trial-micro overall | 0.0866 |
| easy / hard | 0.0970 / 0.0437 |
| seen / unseen | 0.0580 / 0.1045 |
| PCK@0.5·min | 0.0266 |
| AUC（τ 0.1–1.0） | 0.0634 |
| 单帧时延 | 76.2 ms（RTX 4090，batch=1，CPU 方法） |

## 产物

| 文件 | 内容 |
|---|---|
| results.jsonl | 逐 trial 预测点，原图 1920×1080 像素 |
| peaks.jsonl | top-32 峰（原图坐标 + 响应分），NMS 32px |
| metrics.json | v3 test 聚合指标（随包） |

results.jsonl / peaks.jsonl 体积大，不随包；按上面的全量命令重建，再用
evaluation/aggregate.py 聚合。

## 实现要点

1. TM_CCOEFF_NORMED 的响应坐标是模板窗口左上角。三尺度窗口尺寸不同，故每个尺度响应先按
   +(th//2, tw//2) 平移到窗口中心坐标系再逐点取 max，避免尺度相关偏移。
2. 缩放后模板任一维超过图像维则该尺度跳过并计数（summary.json: n_scale_skipped）。
3. 下采样 INTER_AREA、上采样 INTER_LINEAR，与仓库其它分支同族。
4. 坐标全程为原图 1920×1080，无 letterbox 逆变换。
