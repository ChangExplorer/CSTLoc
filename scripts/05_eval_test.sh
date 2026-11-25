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

LOCATOR="$WEIGHTS/e116b_s42/best_model.pth"
TCIA="$WEIGHTS/tcia_v2_s2024/best_fullval.pt"
TAG="final_s2024"
OUT_DIR="$ROOT/evaluation/results"
while [ $# -gt 0 ]; do
  case "$1" in
    --locator) LOCATOR="$2"; shift 2;;
    --tcia) TCIA="$2"; shift 2;;
    --tag) TAG="$2"; shift 2;;
    --out-dir) OUT_DIR="$2"; shift 2;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done
test -f "$LOCATOR" || { echo "ERROR: locator checkpoint not found: $LOCATOR" >&2; exit 3; }
test -f "$TCIA" || { echo "ERROR: TCIA checkpoint not found: $TCIA" >&2; exit 4; }
EXPERT_DIR="$CSTLOC_EXPERT_DIR"
if [ -z "$EXPERT_DIR" ]; then
  if [ -d "$ROOT/assets/j2_expert_pretrain" ]; then EXPERT_DIR="$ROOT/assets/j2_expert_pretrain"
  elif [ -d "$WEIGHTS/j2_expert_pretrain" ]; then EXPERT_DIR="$WEIGHTS/j2_expert_pretrain"
  else EXPERT_DIR="$CODE/checkpoints/j2_expert_pretrain"; fi
fi
test -f "$EXPERT_DIR/head_random.pt" || { echo "ERROR: head_random.pt not found: $EXPERT_DIR" >&2; exit 5; }
export CSTLOC_EXPERT_DIR="$EXPERT_DIR"
export CSTLOC_LOCATOR_EXPERT="$EXPERT_DIR/head_random.pt"
export CSTLOC_LOCATOR_CKPT="$LOCATOR"
export CSTLOC_LOCATOR_MODULATOR="correlation"
export CSTLOC_LOCATOR_USE_ID="1"
export TCIA_BEST_CKPT="$TCIA"
export ABL_TAG="$TAG"
LOCATOR="$(realpath -m "$LOCATOR")"
TCIA="$(realpath -m "$TCIA")"
OUT_DIR="$(realpath -m "$OUT_DIR")"
mkdir -p "$ROOT/evaluation/cache/ablation" "$OUT_DIR"
echo "[05] test eval: tag=$TAG"
cd "$CODE"
"$PY" -u tools/eval_ablation_tcia.py
TAG_COPY="${ABL_TAG:-$TAG}"
if [ -z "$TAG_COPY" ]; then TAG_COPY="final_s2024"; fi
cp "$ROOT/evaluation/cache/ablation/tcia_$TAG_COPY_test_summary.json" "$OUT_DIR/tcia_$TAG_COPY_test_summary.json"
echo "[05] OK: $OUT_DIR/tcia_$TAG_COPY_test_summary.json"
