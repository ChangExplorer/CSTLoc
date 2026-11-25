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

SEED="2024"
CAND="$CODE/diag_out/tcia_v2_train_trials.jsonl"
LOCATOR="$WEIGHTS/e116b_s42/best_model.pth"
TCIA_DIR="$WEIGHTS/tcia_v2_s2024"
TAG="final_s2024"
CAND_SHA="$TCIA_TRAIN_SHA"
while [ $# -gt 0 ]; do
  case "$1" in
    --seed) SEED="$2"; shift 2;;
    --candidates) CAND="$2"; shift 2;;
    --locator) LOCATOR="$2"; shift 2;;
    --ckpt-dir) TCIA_DIR="$2"; shift 2;;
    --tag) TAG="$2"; shift 2;;
    --candidate-sha) CAND_SHA="$2"; shift 2;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done
test -f "$CAND" || { echo "ERROR: run 03_build_candidates.sh first" >&2; exit 3; }
test -f "$LOCATOR" || { echo "ERROR: locator checkpoint not found: $LOCATOR" >&2; exit 4; }
if [ -z "$CAND_SHA" ]; then CAND_SHA="$(sha256sum "$CAND" | awk '{print $1}')"; fi
CAND="$(realpath -m "$CAND")"
LOCATOR="$(realpath -m "$LOCATOR")"
TCIA_DIR="$(realpath -m "$TCIA_DIR")"
mkdir -p "$TCIA_DIR"
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
export TCIA_CKPT_DIR="$TCIA_DIR"
export TCIA_TAG="$TAG"
export TCIA_VAL_TOTAL="${TCIA_VAL_TOTAL:-171587}"
export TCIA_TEST_TOTAL="${TCIA_TEST_TOTAL:-327772}"
export TCIA_TRAIN_TRIALS="$CAND"
export TCIA_TRAIN_SHA="$CAND_SHA"
export TCIA_SEED="$SEED"
echo "[04] train TCIA: seed=$SEED locator=$LOCATOR cand_sha=$CAND_SHA out=$TCIA_DIR"
cd "$CODE"
"$PY" -u tools/tcia_v2_train.py --phase train --seed "$SEED"
test -f "$TCIA_DIR/best_fullval.pt" || { echo "ERROR: best_fullval.pt not produced" >&2; exit 6; }
echo "[04] OK: $TCIA_DIR/best_fullval.pt"
