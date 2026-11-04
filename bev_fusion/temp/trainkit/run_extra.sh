#!/usr/bin/env bash
# scratch: the slower tests that touch train.py (checkpoint helpers, warm start, evaluation adapter); output to $1
cd "$(dirname "$0")/../.." || exit 1
export PYTHONPATH="$(cd .. && pwd)${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
PY="$HOME/miniconda3/envs/gcar/bin/python"
log=${1:-temp/trainkit/extra.log}
: > "$log"
for t in tests/test_gt_cache_targets.py tests/test_velocity_hurdle.py tests/test_end_to_end.py tests/test_runner.py tests/test_radar_slot.py; do
  [ -f "$t" ] || { echo "$t missing" >> "$log"; continue; }
  s=$(date +%s)
  "$PY" "$t" > temp/trainkit/last_extra.out 2>&1
  rc=$?
  echo "$t rc=$rc $(( $(date +%s) - s ))s | $(grep -v -i warn temp/trainkit/last_extra.out | tail -2 | tr '\n' ' ' | cut -c1-200)" >> "$log"
done
echo ALLDONE >> "$log"
