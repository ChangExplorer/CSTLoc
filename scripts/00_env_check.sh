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

echo "[00] environment"
echo "ROOT=$ROOT"
echo "PY=$PY"
echo "DATA_ROOT=$DATA_ROOT"
echo "WEIGHTS=$WEIGHTS"
"$PY" - <<'PY'
import sys
import torch, cv2, numpy, yaml
print("python", sys.version.split()[0])
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
print("opencv", cv2.__version__, "numpy", numpy.__version__)
PY
test -d "$DATA_ROOT/train" || { echo "ERROR: dataset not found: $DATA_ROOT" >&2; exit 2; }
if [ ! -f "$ROOT/infmae/InfMAE.pth" ] && [ ! -f "$WEIGHTS/InfMAE.pth" ]; then
  echo "ERROR: InfMAE.pth not found (put under infmae/ or weights/)" >&2; exit 3
fi
echo "[00] OK"
