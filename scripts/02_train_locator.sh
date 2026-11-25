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
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export PYTHONPATH="$CODE"
export CSTLOC_DATA_ROOT="$DATA_ROOT"
export CSTLOC_WEIGHTS="$WEIGHTS"
INFMAE_CKPT="$CSTLOC_INFMAE_CKPT"
if [ -z "$INFMAE_CKPT" ] && [ -f "$WEIGHTS/InfMAE.pth" ]; then INFMAE_CKPT="$WEIGHTS/InfMAE.pth"; fi
if [ -n "$INFMAE_CKPT" ]; then export CSTLOC_INFMAE_CKPT="$INFMAE_CKPT"; fi

# 发布包把注入式随机专家头放在 assets/ 下；train.py 用 CSTLOC_EXPERT_DIR 解析同名文件。
EXPERT_DIR="$CSTLOC_EXPERT_DIR"
if [ -z "$EXPERT_DIR" ]; then
  if [ -d "$ROOT/assets/j2_expert_pretrain" ]; then EXPERT_DIR="$ROOT/assets/j2_expert_pretrain"
  elif [ -d "$WEIGHTS/j2_expert_pretrain" ]; then EXPERT_DIR="$WEIGHTS/j2_expert_pretrain"
  else EXPERT_DIR="$CODE/checkpoints/j2_expert_pretrain"; fi
fi
export CSTLOC_EXPERT_DIR="$EXPERT_DIR"
test -f "$EXPERT_DIR/head_random.pt" || { echo "ERROR: head_random.pt not found: $EXPERT_DIR" >&2; exit 3; }

CONFIG="$CODE/configs/final/e116b_s42.yaml"
SEED="42"
SAVE_DIR="$WEIGHTS/e116b_s42"
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
