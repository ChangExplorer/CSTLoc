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

OUT="$CODE/diag_out/tcia_train_trials.jsonl"
LOCATOR="$WEIGHTS/locator/best_model.pth"
N_TRIALS="20000"
SEED="42"
while [ $# -gt 0 ]; do
  case "$1" in
    --out) OUT="$2"; shift 2;;
    --locator) LOCATOR="$2"; shift 2;;
    --n-trials) N_TRIALS="$2"; shift 2;;
    --seed) SEED="$2"; shift 2;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done
OUT="$(realpath -m "$OUT")"
LOCATOR="$(realpath -m "$LOCATOR")"
mkdir -p "$(dirname "$OUT")"
test -f "$LOCATOR" || { echo "ERROR: locator checkpoint not found: $LOCATOR" >&2
                        echo "Run scripts/02_train_locator.sh first (or pass --locator)." >&2; exit 3; }
export CSTLOC_LOCATOR_EXPERT="random"
export CSTLOC_LOCATOR_CKPT="$LOCATOR"
export CSTLOC_LOCATOR_MODULATOR="correlation"
export CSTLOC_LOCATOR_USE_ID="1"
echo "[03] build TCIA train candidates from locator=$LOCATOR n=$N_TRIALS seed=$SEED"
cd "$CODE"
"$PY" -u tools/build_tcia_candidates.py --out "$OUT" --locator "$LOCATOR" --n-trials "$N_TRIALS" --seed "$SEED"
echo "[03] OK: $OUT"
