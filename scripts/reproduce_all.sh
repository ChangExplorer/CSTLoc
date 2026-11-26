#!/usr/bin/env bash
set -eo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CODE="$ROOT/code"
PY="$PYTHON"
DATA_ROOT="$CSTLOC_DATA_ROOT"
if [ -z "$PY" ] || [ -z "$DATA_ROOT" ]; then
  echo "Required environment:" >&2
  echo "  export PYTHON=/path/to/python            # python 3.10 + torch (see requirements.txt)" >&2
  echo "  export CSTLOC_DATA_ROOT=/path/to/CSTLoc  # dataset root with train/ val/ test/" >&2
  exit 2
fi
WEIGHTS="$CSTLOC_WEIGHTS"
if [ -z "$WEIGHTS" ]; then WEIGHTS="$ROOT/weights"; fi
export PYTHONPATH="$CODE"
export CSTLOC_DATA_ROOT="$DATA_ROOT"
export CSTLOC_WEIGHTS="$WEIGHTS"

SKIP_TRAIN="$SKIP_TRAIN"
if [ -z "$SKIP_TRAIN" ]; then SKIP_TRAIN="0"; fi
bash "$ROOT/scripts/00_env_check.sh"
bash "$ROOT/scripts/01_prepare_dataset.sh"
"$PY" "$ROOT/scripts/01b_build_test_cache.py"
if [ "$SKIP_TRAIN" != "1" ]; then
  # 全链：自训定位器 → 重建候选 → 训练 TCIA；SKIP_TRAIN=1 时只评测已有权重
  bash "$ROOT/scripts/02_train_locator.sh"
  bash "$ROOT/scripts/03_build_candidates.sh"
  bash "$ROOT/scripts/04_train_tcia.sh"
fi
bash "$ROOT/scripts/05_eval_test.sh"
bash "$ROOT/scripts/06_reproduce_tables.sh"
echo "[all] done"
