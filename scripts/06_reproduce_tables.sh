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

SUM="$ROOT/evaluation/results/tcia_final_test_summary.json"
if [ ! -f "$SUM" ]; then echo "ERROR: run 05_eval_test.sh first ($SUM)" >&2; exit 3; fi
"$PY" - "$SUM" <<'PY'
import json, sys
s = json.load(open(sys.argv[1], encoding="utf-8"))
want = {"overall": 0.5542, "easy": 0.7971, "hard": 0.4773, "seen": 0.5662, "unseen": 0.4618}
print("summary keys:", sorted(s.keys()))
for k in ("overall", "easy", "hard", "seen", "unseen"):
    v = s.get(k)
    print(f"{k}: {v} (paper {want[k]})")
print("NOTE: baseline rows are reproduced by the per-method commands in baselines/README.md; this script checks the main row only.")
PY
