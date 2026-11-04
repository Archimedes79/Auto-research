#!/usr/bin/env bash
# scratch: run the tests that exercise train.py (baseline before / checks after the trainkit move); output to $1 (a log file)
cd "$(dirname "$0")/../.." || exit 1
export PYTHONPATH="$(cd .. && pwd)${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
PY="${PY:-$HOME/miniconda3/envs/gcar/bin/python}"
log=${1:-temp/trainkit/related.log}
: > "$log"
for t in tests/test_lr_groups.py tests/test_grad_scaler.py tests/test_speed_flags.py tests/test_host_overhead.py tests/test_clip_split.py tests/test_freeze_probe.py tests/test_hygiene.py; do
  s=$(date +%s)
  "$PY" "$t" > temp/trainkit/last.out 2>&1
  rc=$?
  echo "$t rc=$rc $(( $(date +%s) - s ))s | $(grep -v -i warn temp/trainkit/last.out | tail -1 | cut -c1-160)" >> "$log"
done
echo ALLDONE >> "$log"
