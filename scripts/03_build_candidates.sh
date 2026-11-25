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

ARCHIVE="$ROOT/dataset/tcia/tcia_v2_train_trials.jsonl.xz"
OUT="$CODE/diag_out/tcia_v2_train_trials.jsonl"
EXPECT="4e4b27a0a41e0b9b659b12a1a5327facf87fc6de62759f98eacba3b8602a5b43"
LOCATOR="$WEIGHTS/e116b_s42/best_model.pth"
N_TRIALS="20000"
SEED="42"
GENERATE="0"
while [ $# -gt 0 ]; do
  case "$1" in
    --from-locator) GENERATE="1"; shift;;
    --unpack) GENERATE="0"; shift;;
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

if [ "$GENERATE" = "1" ]; then
  test -f "$LOCATOR" || { echo "ERROR: locator checkpoint not found: $LOCATOR" >&2; exit 3; }
  export CSTLOC_LOCATOR_EXPERT="random"
  export CSTLOC_LOCATOR_CKPT="$LOCATOR"
  export CSTLOC_LOCATOR_MODULATOR="correlation"
  export CSTLOC_LOCATOR_USE_ID="1"
  echo "[03] rebuild TCIA train candidates from locator=$LOCATOR n=$N_TRIALS seed=$SEED"
  cd "$CODE"
  "$PY" -u tools/build_tcia_candidates.py --out "$OUT" --locator "$LOCATOR" --n-trials "$N_TRIALS" --seed "$SEED"
else
  test -f "$ARCHIVE" || { echo "ERROR: $ARCHIVE missing" >&2; exit 3; }
  echo "[03] unpack TCIA train candidates: $ARCHIVE"
  xz -dc "$ARCHIVE" > "$OUT"
  GOT="$(sha256sum "$OUT" | awk '{print $1}')"
  echo "[03] sha256=$GOT"
  if [ "$GOT" != "$EXPECT" ]; then echo "ERROR: candidate sha mismatch: $GOT != $EXPECT" >&2; exit 4; fi
fi
GOT="$(sha256sum "$OUT" | awk '{print $1}')"
echo "[03] OK: $OUT sha256=$GOT"
