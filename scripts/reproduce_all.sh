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

bash "$ROOT/scripts/00_env_check.sh"
bash "$ROOT/scripts/01_prepare_dataset.sh"
"$PY" "$ROOT/scripts/01b_build_test_cache.py"
bash "$ROOT/scripts/02_train_locator.sh"
bash "$ROOT/scripts/03_build_candidates.sh"
bash "$ROOT/scripts/04_train_tcia.sh"
bash "$ROOT/scripts/05_eval_test.sh"
echo "[all] done"
