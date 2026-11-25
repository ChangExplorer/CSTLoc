#!/usr/bin/env bash
set -eo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CODE="$ROOT/code"
PY="$PYTHON"
if [ -z "$PY" ]; then PY="/home/user/anaconda3/envs/viir/bin/python"; fi
DATA_ROOT="$CSTLOC_DATA_ROOT"
if [ -z "$DATA_ROOT" ]; then DATA_ROOT="/home/user/Code/Datasets/CSTLoc"; fi
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
  # 默认使用发布定位器；TRAIN_LOCATOR=1 时从零训练定位器并重建候选（全链）
  if [ "$TRAIN_LOCATOR" = "1" ]; then
    bash "$ROOT/scripts/02_train_locator.sh"
    bash "$ROOT/scripts/03_build_candidates.sh" --from-locator
  else
    bash "$ROOT/scripts/03_build_candidates.sh"
  fi
  bash "$ROOT/scripts/04_train_tcia.sh"
fi
bash "$ROOT/scripts/05_eval_test.sh"
bash "$ROOT/scripts/06_reproduce_tables.sh"
echo "[all] done"
