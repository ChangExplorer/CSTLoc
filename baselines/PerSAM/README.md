# PerSAM 对照臂（C 臂）工作区

> 状态：**全量 test 完成**（2026-10-23/24，4080；官方 **307,527 对 / 24 实例**）。
> 设计：参考 = RGB 模板（中心 80% 作为 mask，对应模板 margin 1.2 → 目标约 83%）；目标 = IR 场景；
> training-free PerSAM → 最终 mask 质心作为预测点；σ 不用（SAM 自身分数仅记录）。

## 资产

| 项 | 值 |
|---|---|
| 上游代码 | `references/repos/Personalize-SAM`（官方 tarball，MIT；git clone 被墙） |
| SAM ViT-H | `references/weights/sam_vit_h_4b8939.pth`（2.42 GB，sha256 `a7bf3b02…26f262e`，不入库） |
| 环境 | conda `countgd`（torch 2.2.1+cu121；含 `segment-anything`） |
| 运行 | `python run_cstloc.py --split test --out results_test.jsonl`（增量可续跑） |

## 实现要点

1. 模板 embedding 预计算：`set_image(template, mask=中心80%)` → 特征均值 → 归一化（1,666 个模板，一次性）；
2. 场景为主循环：每个 IR 场景只 `set_image` 一次（2,298 次）；
3. 逐对：余弦 sim → 正/负点选择 → 三步级联 mask 解码（官方 PerSAM 流程，含 attn_sim 与 target_embedding）→ 取最终 mask 质心；
4. mask 为空 → 记 miss（empty=1），不做回退（预注册口径）。

## 已验证

- 官方 demo（cat 参考→另两张图）通过：mask 生成、可视化正常；
- 单对可视化检查（bus_025）：PerSAM 分割出**画面左侧的另一辆公交车**，而 GT 目标在右侧 (1675,901) —— 管线正常，失败形态是**实例混淆**（找到同类、选错个体），非代码 bug。

## 成本估算

- 场景编码 2,298 × ~0.25 s ≈ 10 min；模板 embedding 1,666 × ~0.1 s ≈ 3 min；
- 逐对 ~20–30 ms → 全量 307,527 对 ≈ **2–3 h**（4080，单卡）。

## 结果（全量 test，307,527 对 / 24 实例）

| 指标 | PerSAM |
|---|---|
| pair-macro PCK@0.5·max | **0.0493** |
| easy / hard | 0.0631 / 0.0383 |
| seen / unseen | 0.0499 / 0.0233（unseen 含 animal_004 与 truck_006，n=6,955） |
| PCK@0.5·min | 0.0257 |
| AUC（τ 0.1–1.0） | 0.0600 |
| 空 mask 率 | 11.2% |
| 预测落同类干扰物框 | 3.1% |
| 平均 mask 分数 | 0.920 |
| 平均每对耗时 | 31.4 ms（4080） |

**口径说明**：与主表其他臂完全一致，均为 307,527 对（24 实例）。实现时第一遍误用了训练期的
`rgb_usable=false` 过滤（animal_004 夜间可见光模板），漏掉 5,665 对；已单独补跑 animal_004
（55 模板 × 103 帧）并合并为 `results_test_full.jsonl`。animal_004 的模板为夜间近全黑帧，
命中率仅 4.6%，unseen 桶由 animal 与 truck 构成。

**失败分析**：空 mask 仅 11.2%，说明多数情况下能分割出东西；但非空预测中落入同类干扰物框的
只有 3.1%，其余落在画面其它区域——跨模态参考外观（RGB 模板）无法在红外场景中保持身份一致，
分割在错误位置"自信地"收敛。定性可视化（bus_025）中，PerSAM 分割到画面左侧的另一辆车而非 GT
目标。失效以实例判别/跨模态迁移为主，而非候选生成。

**复现**：
```
conda activate countgd
cd baselines/PerSAM
python run_cstloc.py --split test --out results_test.jsonl   # 301,862 对，约 3 h（4080）
```
结果 sha256 见 `baselines/EVIDENCE.md`（results_test.jsonl 目标不入库）。
