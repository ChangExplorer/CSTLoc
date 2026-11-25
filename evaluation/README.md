# evaluation/ — 统一评测与诊断

本目录只保留最终评测链：统一聚合器、标准格式适配器、论文 §3.4 的四个诊断脚本。
逐 trial 的中间结果与大缓存一律放在 `evaluation/cache/`，该目录不入 Git。

## 产物契约

每个被评系统输出一个 `results.jsonl`，每行一个 JSON：

```json
{"inst":"car_103","tmpl_mod":"rgb","t":6180,"scene_mod":"ir","f":5610,"pred":[x,y]}
```

- 坐标为**场景图原始像素**（1920×1080），不是 letterbox 坐标。
- 只存预测，不存 GT、难度、类别或 seen/unseen；这些标签由聚合时从数据集注解挂接。
- 数据集根由 `CSTLOC_DATA_ROOT` 指定，默认 `/home/user/Code/Datasets/CSTLoc`；`rgb_usable=false` 的实例自动排除。

## 聚合

```bash
python evaluation/aggregate.py \
  --results evaluation/results/tcia_final_s2024.results.jsonl \
  --split test \
  --out evaluation/results/tcia_final_s2024.metrics.json
```

最终系统的评测由 `scripts/05_eval_test.sh` 完成；它把 summary 写到
`evaluation/results/tcia_final_s2024_test_summary.json`，并把逐 trial 结果留在
`evaluation/results/`。

## 指标（论文 §2.1 的 9 项）

1. PCK@0.5·max：主口径，预测点与 GT 中心距离小于 `0.5·max(h,w)`。
2. PCK@0.5·min：更严格口径，阈值取 `0.5·min(h,w)`。
3. 阈值扫描 AUC：阈值 0.1–1.0，10 个步长。
4. easy/hard 分层：场景是否存在同类干扰物。
5. 实例 rank-1：hard 子集，预测点落入目标实例框记 1，落入同类干扰物框或任一实例框外记 0。
6. 时间间隔效应：按各实例最大 `|f-t|` 归一化后分 10 档。
7. 失效模式二分类：实例混淆（落入同类干扰物框）vs 背景/漏检（其余未命中）。
8. seen/unseen 分组。
9. 单帧时延：单张 RTX 4090，batch=1，含读图、预处理、前向与后处理。

聚合口径：默认 pair-macro（先模板内平均、再模板等权），括号内另报 trial-micro（全部 trial 等权）。
两者定义、公式与参考文献见论文 §2.1。

## 诊断（论文 §3.4）

```bash
python evaluation/diagnostics/diagnostics_v3.py
```

输入（本地缓存，不入 Git）：

- `evaluation/cache/v3_test_diag/main_results/results_full_s{42,123,2024}.jsonl`
- `evaluation/cache/v3_test_diag/main_results/results_locator_s2024.jsonl`
- `evaluation/cache/v3_comparison/{persam,ncc,roma,dinov2,grounding}.results.jsonl`
- 数据集 test 注解（真框与同类干扰物框）

输出 `evaluation/cache/diagnostics_v3/diagnostics_v3.json`。论文使用的字段：

| 字段 | 含义 |
|---|---|
| `oracle@K` | top-K 候选内存在 PCK 命中峰的配对比例 |
| `rank1_instance` | hard 子集实例 rank-1 |
| `fail_instance_hard` / `fail_background_hard` | hard 失败的实例混淆 / 背景漏检占比 |
| `time_curve` | 归一化时间 10 分位 PCK 曲线 |
| `k_curve` | 同类干扰物数量分档 |

图脚本在 `evaluation/diagnostics/`：`select_qual_candidates.py` 选定性对，
`make_fig_qual.py`、`make_fig_diag.py` 分别生成定性图与诊断图（输出到
`evaluation/diagnostics/figures/`）。

**注意**：`aggregate.py` 中的 `instance_rank1_hard_provisional` 与
`failure_taxonomy_provisional` 是早期临时口径（中心点/三分类），只作历史兼容；
论文与最终诊断使用 `diagnostics_v3.py` 的 `rank1_instance`、`fail_instance_hard`、
`fail_background_hard`。

## 最终数字（v3 test，327,772 配对）

| 系统 | overall | hard | 备注 |
|---|---|---|---|
| 本文 Full（三种子均值） | 0.5542 | 0.4773 | trial-micro 0.6917；seed2024 0.5543 |
| 纯定位器 | 0.4499 | 0.3823 | 固定 e116b_s42 |
| Grounding DINO | 0.2116 | 0.1902 | |
| DINOv2 检索 | 0.2071 | 0.1783 | |
| RoMa | 0.1745 | 0.1377 | |
| NCC | 0.0633 | 0.0437 | |
| PerSAM | 0.0440 | 0.0354 | |

候选上限：PCK-oracle@20 = 0.8513（hard 0.8240）。
