#!/usr/bin/env bash
set -eo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/releases/CSTLoc"
rm -rf "$OUT"
mkdir -p "$OUT"
cd "$ROOT"

# 顶层说明
cp README.md CITATION.cff requirements.txt "$OUT/"
cat > "$OUT/LICENSE" <<'EOF'
Copyright (c) 2026 Chang Yao. All rights reserved.

License terms are to be finalized. This release is provided for academic
review and reproduction of the associated manuscript. Contact the author
before redistribution.
EOF

# 代码（只保留最终闭包）
rsync -a code "$OUT/" \
  --exclude '_archive/' --exclude 'checkpoints/' --exclude 'diag_out/' \
  --exclude 'logs/' --exclude '__pycache__/' --exclude '*.pyc'

# 注入式随机专家头（5MB；release 训练/评测都依赖）
rsync -a assets "$OUT/" --exclude '__pycache__/' --exclude '*.pyc'

# 数据划分与 TCIA 候选
mkdir -p "$OUT/dataset/tcia"
cp dataset/split_v3.json "$OUT/dataset/"
cp dataset/tcia/tcia_v2_train_trials.jsonl.xz dataset/tcia/tcia_v2_train_trials.sha256 "$OUT/dataset/tcia/"

# 评测与诊断
rsync -a evaluation "$OUT/" \
  --exclude 'cache/' --exclude 'results/' --exclude '_archive/' \
  --exclude '__pycache__/' --exclude '*.pyc'

# 五个对照方法（只源码与说明）
rsync -a baselines "$OUT/" \
  --exclude '_archive/' --exclude '*.jsonl' --exclude '*.pt' --exclude '*.pth' \
  --exclude 'shards/' --exclude 'smoke_a/' --exclude 'smoke_b/' \
  --exclude 'vendor/' --exclude '*.log' --exclude '__pycache__/' --exclude '*.pyc'

# 精简文档
mkdir -p "$OUT/docs"
cp docs/权重与数据清单.md "$OUT/docs/"
cp docs/可引用数字清单.md "$OUT/docs/"

# 复现脚本
rsync -a --exclude '__pycache__' --exclude '*.pyc' scripts "$OUT/"

cat > "$OUT/README.md" <<'EOF'
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
EOF

cat > "$OUT/EXPECTED_RESULTS.md" <<'EOF'
# 预期结果（v3 test，327,772 配对）

| 系统 | overall | easy | hard | seen | unseen | PCKmin | AUC | 时延/ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Full（三种子均值） | 0.5542 | 0.7971 | 0.4773 | 0.5662 | 0.4618 | 0.4483 | 0.5001 | 120.8 |
| 纯定位器 | 0.4499 | 0.6332 | 0.3823 | 0.4724 | 0.2761 | 0.3367 | 0.4064 | — |
| Grounding DINO | 0.2116 | 0.3053 | 0.1902 | 0.2277 | 0.0876 | 0.1740 | 0.2225 | 114.5 |
| DINOv2 检索 | 0.2071 | 0.3190 | 0.1783 | 0.2015 | 0.2504 | 0.1053 | 0.1970 | 58.9 |
| RoMa | 0.1745 | 0.2558 | 0.1377 | 0.1768 | 0.1563 | 0.1146 | 0.1730 | 749.3 |
| NCC | 0.0633 | 0.0970 | 0.0437 | 0.0580 | 0.1045 | 0.0266 | 0.0634 | 76.2 |
| PerSAM | 0.0440 | 0.0640 | 0.0354 | 0.0414 | 0.0639 | 0.0206 | 0.0540 | 339.0 |

Full 的 trial-micro overall = 0.6917；三种子为 0.5527 / 0.5557 / 0.5543。

诊断：top-20 候选 PCK-oracle = 0.8513（hard 0.8240）；hard 命中 54.0%，
其中实例混淆 36.1%、背景漏检 10.0%；时间归一化从短到长，Full 衰减 7.0pp，
纯定位器衰减 13.1pp。
EOF

cat > "$OUT/.gitignore" <<'EOF'
__pycache__/
*.pyc
code/diag_out/
code/logs/
code/checkpoints/
evaluation/cache/
evaluation/results/
weights/
infmae/
*.jsonl
EOF

cd "$OUT"
find . -type f ! -name CHECKSUMS -print0 | sort -z | xargs -0 sha256sum > CHECKSUMS
echo "[build_release] wrote $OUT"
du -sh "$OUT"
