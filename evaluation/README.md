# evaluation/ — 统一评测聚合

本目录保留最终评测链：统一聚合器、标准格式适配器。逐 trial 的中间结果与大缓存
一律放在 `evaluation/cache/`，该目录不入 Git。

> 论文 §3.4 的诊断脚本（oracle 上限、失效分类、时间衰减曲线等）是作者内部证据工具，
> 依赖内部缓存，未包含在发布仓中；论文报告的诊断数字见 README 结果图。

## 产物契约

每个被评系统输出一个 `results.jsonl`，每行一个 JSON：

```json
{"inst":"car_103","tmpl_mod":"rgb","t":6180,"scene_mod":"ir","f":5610,"pred":[x,y]}
```

- 坐标为**场景图原始像素**（1920×1080），不是 letterbox 坐标。
- 只存预测，不存 GT、难度、类别或 seen/unseen；这些标签由聚合时从数据集注解挂接。
- 数据集根由 `CSTLOC_DATA_ROOT` 指定（必填，无默认值）；`rgb_usable=false` 的实例自动排除。

## 聚合

```bash
python evaluation/aggregate.py \
  --results evaluation/results/tcia_final.results.jsonl \
  --split test \
  --out evaluation/results/tcia_final.metrics.json
```

最终系统的评测由 `scripts/05_eval_test.sh` 完成；它把 summary 写到
`evaluation/results/tcia_final_test_summary.json`。如需标准格式的逐 trial
`results.jsonl`（供 `aggregate.py` 聚合），用 `code/tools/tcia_eval.py` 的
`run_split(..., write_scores=True)` 落盘逐实例分数后经
`evaluation/adapters/tcia_to_results.py` 转换。

## 指标

`aggregate.py` 输出（除时延外口径见论文 §2.1）：

1. PCK@0.5·max：主口径，预测点与 GT 中心距离小于 `0.5·max(h,w)`。
2. PCK@0.5·min：更严格口径，阈值取 `0.5·min(h,w)`。
3. 阈值扫描 AUC：阈值 0.1–1.0，10 个步长。
4. easy/hard 分层：场景是否存在同类干扰物。
5. seen/unseen 分组。

聚合口径：默认 pair-macro（先模板内平均、再模板等权），括号内另报 trial-micro
（全部 trial 等权）。两者定义、公式与参考文献见论文 §2.1。

单帧时延（论文报告值：RTX 4090，batch=1，含读图/预处理/前向/后处理）由各方法
自行测量，见 `baselines/` 各自 README。

`aggregate.py` 中带 `_provisional` 后缀的字段（实例 rank-1、失效分类）是早期
临时口径，仅供参考；论文使用的最终诊断数字由内部诊断脚本计算（未随发布仓提供）。

## 最终数字（v3 test，327,772 配对）

| 系统 | overall | hard | 备注 |
|---|---|---|---|
| 本文 Full（三种子均值） | 0.5542 | 0.4773 | trial-micro 0.6917；最佳单种子 0.5543 |
| 纯定位器 | 0.4499 | 0.3823 | 冻结发布定位器 |
| Grounding DINO | 0.2116 | 0.1902 | |
| DINOv2 检索 | 0.2071 | 0.1783 | |
| RoMa | 0.1745 | 0.1377 | |
| NCC | 0.0633 | 0.0437 | |
| PerSAM | 0.0440 | 0.0354 | |

候选上限：PCK-oracle@20 = 0.8513（hard 0.8240）。
