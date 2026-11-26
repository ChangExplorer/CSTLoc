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

SEED="0"
CAND="$CODE/diag_out/tcia_train_trials.jsonl"
LOCATOR="$WEIGHTS/locator/best_model.pth"
TCIA_DIR="$WEIGHTS/tcia"
TAG="final"
while [ $# -gt 0 ]; do
  case "$1" in
    --seed) SEED="$2"; shift 2;;
    --candidates) CAND="$2"; shift 2;;
    --locator) LOCATOR="$2"; shift 2;;
    --ckpt-dir) TCIA_DIR="$2"; shift 2;;
    --tag) TAG="$2"; shift 2;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done
test -f "$CAND" || { echo "ERROR: run 03_build_candidates.sh first" >&2; exit 3; }
test -f "$LOCATOR" || { echo "ERROR: locator checkpoint not found: $LOCATOR" >&2; exit 4; }
CAND="$(realpath -m "$CAND")"
LOCATOR="$(realpath -m "$LOCATOR")"
TCIA_DIR="$(realpath -m "$TCIA_DIR")"
mkdir -p "$TCIA_DIR"
export CSTLOC_LOCATOR_EXPERT="random"
export CSTLOC_LOCATOR_CKPT="$LOCATOR"
export CSTLOC_LOCATOR_MODULATOR="correlation"
export CSTLOC_LOCATOR_USE_ID="1"
export TCIA_CKPT_DIR="$TCIA_DIR"
export TCIA_TAG="$TAG"
export TCIA_VAL_TOTAL="${TCIA_VAL_TOTAL:-171587}"
export TCIA_TEST_TOTAL="${TCIA_TEST_TOTAL:-327772}"
export TCIA_TRAIN_TRIALS="$CAND"
export TCIA_SEED="$SEED"
echo "[04] train TCIA: seed=$SEED locator=$LOCATOR out=$TCIA_DIR"
cd "$CODE"
"$PY" -u tools/tcia_train.py --phase train --seed "$SEED"
test -f "$TCIA_DIR/best_fullval.pt" || { echo "ERROR: best_fullval.pt not produced" >&2; exit 6; }
echo "[04] OK: $TCIA_DIR/best_fullval.pt"
