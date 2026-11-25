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

echo "[01] dataset check: $DATA_ROOT"
"$PY" - "$DATA_ROOT" "$ROOT/dataset/split_v3.json" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1]); split = Path(sys.argv[2])
for s in ("train", "val", "test"):
    p = root/s
    assert p.is_dir(), p
    insts = sorted(x.name for x in p.iterdir() if x.is_dir())
    print(f"{s}: {len(insts)} instances")
assert split.exists(), split
obj = json.loads(split.read_text(encoding="utf-8"))
print("split_v3 keys:", sorted(obj.keys())[:8])
print("OK")
PY
