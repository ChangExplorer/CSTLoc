"""Shared filesystem paths and small IO helpers.

The dataset root always comes from the CSTLOC_DATA_ROOT environment variable;
there are no machine-specific defaults anywhere in this package.
"""
import json
import os
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]          # code/
OUT = CODE / 'diag_out'                              # generated artifacts (gitignored)


def data_root() -> Path:
    """Dataset root from CSTLOC_DATA_ROOT (must contain train/ val/ test/)."""
    v = os.environ.get('CSTLOC_DATA_ROOT')
    if not v:
        raise SystemExit(
            'CSTLOC_DATA_ROOT is not set. Point it at the dataset root, e.g.\n'
            '  export CSTLOC_DATA_ROOT=/path/to/CSTLoc   # contains train/ val/ test/')
    return Path(v)


def weights_dir() -> Path:
    """Released checkpoints directory (locator + TCIA + InfMAE)."""
    return Path(os.environ.get('CSTLOC_WEIGHTS', str(CODE.parent / 'weights')))


def write_jsonl(path, rows, mode='w'):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, mode, encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f'WRITE {path.name}: {len(rows)} rows', flush=True)


def read_jsonl(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(l) for l in f]
