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
if [ -z "$CSTLOC_INFMAE_CKPT" ] && [ ! -f "$ROOT/infmae/InfMAE.pth" ]; then
  echo "ERROR: InfMAE checkpoint not found: put it at infmae/InfMAE.pth or set CSTLOC_INFMAE_CKPT" >&2
  echo "       (download instructions: README.md, "Pretrained backbones")" >&2; exit 3
fi
echo "[00] OK"
