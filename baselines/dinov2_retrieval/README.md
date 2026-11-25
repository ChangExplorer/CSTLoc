# baselines/dinov2_retrieval/ —— 臂③ DINOv2 特征检索 + 滑窗

> **执行状态（2026-10-16）：已执行（leader 复核后放行，红线条款的「不自行继续」约束 executor）**。
> test 全量只跑一次：307,527 trials / 172 s，**对宏 PCK@0.5·max = 0.2032**（发布线 0.4680，−26.48pp；
> **未命中红线 0.4580**）。G0 同模态同源自检 A = 0.0009 / 协议配平 A′ = 0.0005
> → **同模态同源检索同样失败**，即瓶颈在「检索范式（全局均值描述子 + patch 级余弦）」而非模态鸿沟。
> 详见 `docs/reports/实验结果23.md` §5。

## 环境

- 解释器：`/home/user/ananconda3/envs/viir/bin/python`（实际路径 `/home/user/anaconda3/envs/viir/bin/python`）
- 主干：`timm` `vit_large_patch14_dinov2.lvd142m`，`pretrained=True`（权重在本机 HF 缓存
  `~/.cache/huggingface/hub/models--timm--vit_large_patch14_dinov2.lvd142m`，离线可用），
  `img_size=224, dynamic_img_size=True`（允许非 224 输入，pos-embed 插值）。
- GPU：RTX 4090（`CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1`）。

## 命令

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1

# 全量 test（307,527 trials，含 G0 同模态同源对照；实例级断点续跑）
python baselines/dinov2_retrieval/run_cstloc.py --tag test

python evaluation/aggregate.py --results baselines/dinov2_retrieval/b23_dinov2_test_results.jsonl \
    --peaks baselines/dinov2_retrieval/b23_dinov2_test_peaks.jsonl --split test

python baselines/dinov2_retrieval/measure_latency.py
```

## 产物

| 文件 | 内容 |
|---|---|
| `b23_dinov2_test_results.jsonl` | `{"inst","tmpl_mod","t","scene_mod","f","pred":[x,y]}`，**原图 1920×1080** |
| `b23_dinov2_test_peaks.jsonl` | top-32 峰（原图坐标 + 余弦分） |
| `b23_dinov2_test_trials.jsonl` | 逐 trial 底稿唯一真源（含 `n_peaks`） |
| `b23_dinov2_frame_{inst}_{首帧:06d}.pt` | 帧级特征图 fp32 `{'frames':[...], 'feats':(F,1024,26,45)}` |
| `b23_dinov2_test_g0.json` | G0 同模态同源对照逐帧证据 |
| `b23_dinov2_test_progress.json` | 已完成实例（断点续跑单元） |
| `latency.json` | 30 帧时延 |

## 管线口径与偏离

1. **输入尺寸**：raw IR → `_letterbox(384×640)`（与发布线同口径）→ **零填充到 392×644**
   （ViT patch14 要求边长整除 14；384%14=6、640%14=10）。零填充不重采样、内容几何不变，
   特征图裁回**有效区 26×45**（patch 行 1..26，中心 y_lb=21..371 落在内容区 y_lb∈[12,372)；
   行 0 中心 y_lb=7 落在 letterbox 上黑边（pad=12）内、预测必越界 orig y=−15，事前按几何剔除；
   列 0..44 中心 x_lb=7..623 全在内容区）。**这是相对任务书字面
   「letterbox 384×640 整图一次前向」的必要偏离**（timm `dynamic_img_size` 下两个维度都必须整除
   patch；填充后右/下边界 patch 含黑边，但该列/行已被裁掉，仅通过 attention 对内部 token 有轻微影响）。
2. **层选择**：任务书未指定 → 事前固定为**最后一个 block（`n=[23]`）**，不做层扫描（避免事后择优）。
3. **模板特征**：`get_intermediate_layers(n=[23], reshape=False)` → (1,257,1024) → 对全部 257 token
   （CLS+256 patch）取 mean（= 任务书「257-token GAP」，与仓库 `intermediate_gap` 同 token 口径）→
   L2 归一化。
4. **场景特征**：`reshape=True` → 每位置 1024-d 向量 L2 归一化；相关 = 矩阵乘
   `(T,1024)@(1024,1170)`（等价 1×1 卷积）→ (T,26,45) 得分图 → argmax。
5. **峰粒度**：patch 步长 14（letterbox）= **42 原图像素**，故 NMS 32px 在 patch 网格上只有 (0,0)
   抑制偏移（相邻 patch 互不抑制，由几何自动导出，`suppress_offsets()`）；top-32 = 前 32 名 patch。
   定位粒度因此为 42 px 量级——对 test 小目标（obj 最大边常 < 25 px、PCK 阈值 ≈ 0.5·max(obj) ≈
   9–13 px）构成**范式级上限**，报告如实登记，不做网格插值（避免改预设设计）。
6. **坐标**：patch (r,c) 中心 letterbox = (14c+7, 14(r+1)+7) → 原图 `(x*3, (y−12)*3)`；
   预测范围实测 x∈[21,1869]、y∈[27,1077]，全在原图内。
7. **G0 口径**（同模态同源对照，分离「模态鸿沟」与「检索范式」两因子）：
   - **A（任务书字面）**：模板 = **同一帧** IR 图在 GT 处的 224 裁窗（同模态同源），逐帧 1 模板 → 逐帧 PCK；
   - **A′（协议配平）**：把 A 的同一模板向量用于该帧全部合格 trial（模板与 trial 的 t 无关）→
     trial 加权，**与主管线 C 同 trial 集、同相关式、同 argmax/NMS 口径，可直接比**；
   - **C = 主管线**（跨模态 RGB 模板）trial PCK，由 `evaluation/aggregate.py` 给。
   实测：**A = 0.0009、A′ = 0.0005、C = 0.2032** → 任务书预注册的「同模态同源应 ≫ 跨模态」**不成立**
   （同模态反而更低），因子分离结论 = **范式瓶颈，非模态鸿沟**。

## 机制诊断（同一 run 的派生物，`b23_dinov2_test_mech.json`）

| 观测量 | 值 | 读法 |
|---|---|---|
| 模板描述子实例内两两余弦均值 | **0.8422** | 257-token GAP 描述子在模板之间高度共线 → 得分图随 t 变化有限 |
| 使用到的预测位置 | 1170 / 1170（26×45 全覆盖） | **未**退化到少数位置 |
| 边界带（有效区最外一圈 patch）占比 | **0.1581**（均匀 0.1179 → 相对富集 **1.34×**） | 预测偏向边界/低上下文 patch，与「全局均值描述子最接近低上下文 patch」一致 |
| 同帧内平均不同预测数 | 12.06 | 描述子有模板敏感性，但远不足以定位 |
| 同帧众数预测占比 | 0.4724 | 近半 trial 落在同一位置 |
| 257- vs 256-token 池化预测差异 | 0.3694% of trials（pair_macro 同 0.2032） | 池化选择**不是**本臂结果的决定因素 |

## 追加偏离与实测修正（执行后登记）

| # | 条目 | 说明 |
|---|---|---|
| D3-7 | **任务书「257-token GAP」的落地修正** | 本机 timm 1.0.26 的 `get_intermediate_layers(reshape=False)` **只返回 256 个 patch token、不含 CLS**（实测：224→256、392×644→1288），而仓库 `intermediate_gap` 的 docstring 写「含 CLS，257 个」与实测不符（**仓库侧文档偏差，登记不改仓库**）。本臂按任务书字面口径，用 `return_prefix_tokens=True` 显式取回 CLS 再与 patch 拼接后取 mean = **257-token GAP（主口径）**；同时算 **256-token patch-only**（= 仓库 `intermediate_gap` 在当前 timm 下的实际行为）作为敏感度行：两者 pair_macro **同为 0.2032**、仅 0.37% 的 trial 预测不同 → 池化选择不影响结论 |
| D3-8 | **G0 的 B 变体作废** | 原设计的「模板帧 t 的 IR 图裁窗」不可行：**test 的模板帧集合与 search 帧集合不相交**（如 animal_004 模板帧 0,60,120… / search 帧 30,90,150…），模板帧根本没有 IR search 图。故改为 A（帧内自源）+ A′（同一模板用于该帧全部 trial，trial 加权），二者与 C 同 trial 集可比 |
