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
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export PYTHONPATH="$CODE"
export CSTLOC_DATA_ROOT="$DATA_ROOT"
export CSTLOC_WEIGHTS="$WEIGHTS"
INFMAE_CKPT="$CSTLOC_INFMAE_CKPT"
if [ -z "$INFMAE_CKPT" ] && [ -f "$ROOT/infmae/InfMAE.pth" ]; then INFMAE_CKPT="$ROOT/infmae/InfMAE.pth"; fi
if [ -n "$INFMAE_CKPT" ]; then export CSTLOC_INFMAE_CKPT="$INFMAE_CKPT"; fi

# 注入式随机专家头由 IdentityExpert 按固定 seed 生成，无外部文件依赖。

CONFIG="$CODE/configs/final/locator.yaml"
SEED="0"
SAVE_DIR="$WEIGHTS/locator"
while [ $# -gt 0 ]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2;;
    --seed) SEED="$2"; shift 2;;
    --save-dir) SAVE_DIR="$2"; shift 2;;
    --data-root) DATA_ROOT="$2"; export CSTLOC_DATA_ROOT="$DATA_ROOT"; shift 2;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done
CONFIG="$(realpath -m "$CONFIG")"
SAVE_DIR="$(realpath -m "$SAVE_DIR")"
mkdir -p "$SAVE_DIR"
echo "[02] train locator: config=$CONFIG seed=$SEED save_dir=$SAVE_DIR"
cd "$CODE"
"$PY" -u engine/train.py --config "$CONFIG" --data-root "$DATA_ROOT" --save-dir "$SAVE_DIR" --seed "$SEED"
test -f "$SAVE_DIR/best_model.pth" || { echo "ERROR: best_model.pth not produced" >&2; exit 4; }
echo "[02] OK: $SAVE_DIR/best_model.pth"
