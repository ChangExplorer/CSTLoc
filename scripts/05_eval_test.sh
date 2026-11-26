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

LOCATOR="$WEIGHTS/locator/best_model.pth"
TCIA="$WEIGHTS/tcia/best_fullval.pt"
TAG="final"
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
export CSTLOC_LOCATOR_EXPERT="random"
export CSTLOC_LOCATOR_CKPT="$LOCATOR"
export CSTLOC_LOCATOR_MODULATOR="correlation"
export CSTLOC_LOCATOR_USE_ID="1"
export TCIA_BEST_CKPT="$TCIA"
LOCATOR="$(realpath -m "$LOCATOR")"
TCIA="$(realpath -m "$TCIA")"
OUT_DIR="$(realpath -m "$OUT_DIR")"
mkdir -p "$OUT_DIR"
export EVAL_TAG="$TAG"
export EVAL_OUT="$OUT_DIR/tcia_${TAG}_test_summary.json"
echo "[05] test eval: tag=$TAG"
cd "$CODE"
"$PY" -u tools/evaluate_tcia.py
echo "[05] OK: $EVAL_OUT"
