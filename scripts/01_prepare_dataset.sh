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

echo "[01] dataset check: $DATA_ROOT"
"$PY" - "$DATA_ROOT" <<'PY'
import sys
from pathlib import Path
root = Path(sys.argv[1])
for s in ("train", "val", "test"):
    p = root/s
    assert p.is_dir(), p
    insts = sorted(x.name for x in p.iterdir() if x.is_dir())
    print(f"{s}: {len(insts)} instances")
print("OK")
PY
