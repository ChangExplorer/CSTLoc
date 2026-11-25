#!/usr/bin/env bash
set -eo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/releases/CSTLoc"
test -d "$OUT" || { echo "ERROR: $OUT missing; run scripts/build_release.sh" >&2; exit 2; }
bad=0
for p in paper weights code/checkpoints code/diag_out code/logs evaluation/cache evaluation/results baselines/_archive code/_archive code/tools/_archive .git; do
  if [ -e "$OUT/$p" ]; then echo "ERROR: forbidden path in release: $p" >&2; bad=1; fi
done
for f in "$OUT"/README.md "$OUT"/requirements.txt "$OUT"/EXPECTED_RESULTS.md "$OUT"/CHECKSUMS "$OUT"/scripts/reproduce_all.sh "$OUT"/code/configs/final/e116b_s42.yaml "$OUT"/code/configs/final/tcia_v2_s2024.yaml "$OUT"/dataset/split_v3.json "$OUT"/dataset/tcia/tcia_v2_train_trials.jsonl.xz "$OUT"/assets/j2_expert_pretrain/head_random.pt; do
  if [ ! -f "$f" ]; then echo "ERROR: missing $f" >&2; bad=1; fi
done
if find "$OUT" -type f \( -name '*.pt' -o -name '*.pth' -o -name '*.jsonl' \) \
    | grep -v '^'"$OUT"'/assets/j2_expert_pretrain/head_random.pt$' | grep -q .; then
  echo "ERROR: large weight/trial files present" >&2; bad=1
fi
if [ "$bad" -ne 0 ]; then exit 3; fi
echo "[check_release] OK"
du -sh "$OUT"
find "$OUT" -type f | wc -l
