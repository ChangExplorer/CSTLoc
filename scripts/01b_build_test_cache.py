#!/usr/bin/env python
"""Build the deterministic DINOv2 template-feature cache for a split (default: test).

The cache is derived from the dataset only (no checkpoints needed) and is required
by the evaluator. Run once after 01_prepare_dataset.sh; takes about a minute on one GPU.
Usage: python scripts/01b_build_test_cache.py [split] [--force]
"""
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parents[1] / 'code'
sys.path.insert(0, str(CODE))

from tools.template_cache import build  # noqa: E402

if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    split = args[0] if args else 'test'
    build(split, force='--force' in sys.argv)
