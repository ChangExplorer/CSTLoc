# CSTLoc

CSTLoc 是跨时空可见光—红外**实例级目标定位**基准与最终系统。给定目标实例在某一时刻的
可见光切片，在另一时空获取的红外整帧场景中定位该实例；场景中可能存在多个同类干扰物。

本仓库是最终系统的可复现包，包含：

- 冻结双骨干定位器：DINOv2 ViT-L（可见光模板）+ InfMAE ViT-B（红外场景）→ 三尺度 CorrLayer → UNet 热图；
- TCIA 身份适配器：top-20 候选上学习候选级身份嵌入，用同场景同类干扰与错误模板对比训练；
- CSTLoc 数据划分、评测聚合器、论文 §3.4 的四个诊断脚本；
- 五个对照方法的适配源码。

论文最终结果：v3 test 327,772 配对，pair-macro PCK@0.5·max = 0.5542（三种子均值；
seed 2024 为 0.5543）。

## 1. 目录

```
code/        定位器、TCIA、数据、损失、训练与评测入口
scripts/     00–06 复现脚本
evaluation/  统一聚合器、适配器与诊断脚本
baselines/   五个对照方法源码
dataset/     split_v3.json 与 TCIA 训练候选（xz）
docs/        权重/数据清单与可引用数字
```

## 2. 环境

- Linux / WSL2，PyTorch 2.5.1+cu121，Python 3.10，单卡 RTX 4090（24GB）。
- `pip install -r requirements.txt`；torch/torchvision 用官方 cu121 索引。
- 系统内存建议 ≥15GB；训练与评测串行。
- 多卡机器用 `CUDA_VISIBLE_DEVICES=<4090 的序号>` 选择 4090；脚本默认
  `CUDA_DEVICE_ORDER=PCI_BUS_ID`，避免 CUDA 与 nvidia-smi 枚举顺序不一致。

## 3. 外部文件

本包不含数据集、预训练权重与训练好的 checkpoint。放置如下：

```
CSTLoc/
  infmae/InfMAE.pth                         # InfMAE ViT-B，1.15GB
  weights/
    e116b_s42/best_model.pth                # 固定定位器；复现发布结果必须提供
    tcia_v2_s2024/best_fullval.pt           # 只用已有权重做 L1 时放置
  assets/j2_expert_pretrain/head_random.pt  # 注入式随机专家头，5MB（随发布包提供）
  <dataset root>/train|val|test              # 由 CSTLOC_DATA_ROOT 指定
```

sha256 与数据规模见 `docs/权重与数据清单.md`。数据集获取：论文接收前联系
yaochang_@nudt.edu.cn；接收后补网盘链接与校验值。

## 4. 复现（定位器固定）

```bash
export CSTLOC_DATA_ROOT=/path/to/CSTLoc
export PYTHON=/path/to/python

bash scripts/00_env_check.sh
bash scripts/01_prepare_dataset.sh
"$PYTHON" scripts/01b_build_test_cache.py     # 生成测试模板缓存（约 30s，5.6GB；不入 Git）
bash scripts/03_build_candidates.sh       # 解压并校验 TCIA 候选（固定定位器）
bash scripts/04_train_tcia.sh             # 固定 e116b_s42，从零重训 TCIA seed2024
bash scripts/05_eval_test.sh              # v3 test 全量
bash scripts/06_reproduce_tables.sh
```

发布系统的定位器固定为 e116b_s42（随外部权重提供）；从零复现训练的是 TCIA，其输入候选已随包提供。由于 GPU 训练本身的非确定性，从零训练的 checkpoint 不承诺与发布权重逐位一致；主表由 weights/ 中的最终权重评测复现，指标逐位一致。

### 4.1 全链从零训练（v3，定位器 seed42 + TCIA seed2024）

只依赖发布包代码、v3 数据集与公开预训练权重，不依赖上面的 weights/：

```bash
export CSTLOC_DATA_ROOT=/path/to/CSTLoc
export PYTHON=/path/to/python
export CSTLOC_INFMAE_CKPT=/path/to/InfMAE.pth

bash scripts/01_prepare_dataset.sh
"$PYTHON" scripts/01b_build_test_cache.py
bash scripts/02_train_locator.sh --seed 42            # 定位器 seed42
bash scripts/03_build_candidates.sh --from-locator    # 用新定位器重建 20k TCIA 候选
bash scripts/04_train_tcia.sh --seed 2024             # TCIA seed2024
bash scripts/05_eval_test.sh --tag from_scratch_v3
```

候选重建需要定位器权重 `weights/e116b_s42/best_model.pth`（由 02 产出），不需要旧的 CMI 特征缓存。

本机 2026-11-25 实测：从零全链（seed42/2024）v3 test overall = 0.5426（hard 0.4691），与发布权重 0.5543 相差 -1.17pp；主表仍以发布权重评测为准，4.1 用于验证训练链可跑通。

若已有权重、只想评测：

```bash
SKIP_TRAIN=1 bash scripts/reproduce_all.sh
```

## 5. 预期结果

| 指标 | 值 |
|---|---|
| overall（pair-macro PCK@0.5·max） | 0.5542 |
| trial-micro overall | 0.6917 |
| easy / hard | 0.7971 / 0.4773 |
| seen / unseen | 0.5662 / 0.4618 |
| PCK@0.5·min | 0.4483 |
| 单帧时延 | 120.8 ms（RTX 4090，batch=1） |

逐项见 `EXPECTED_RESULTS.md`。定位器与 TCIA 分开训练再串联；模型选择只用 val，
test 只用于最终报告。

## 6. 对照方法

`baselines/` 中五个方法各自说明依赖、权重与运行命令；统一输出 `results.jsonl` 后用
`evaluation/aggregate.py` 聚合。对照数字不参与最终系统训练。

## 7. 引用

见 `CITATION.cff`。
