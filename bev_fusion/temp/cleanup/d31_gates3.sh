#!/bin/bash
cd /mnt/c/Auto-research/bev_fusion || exit 1
O=/tmp/d31_gates3_mine; rm -rf "$O"; mkdir -p "$O"
PY=~/miniconda3/envs/gcar/bin/python
export PYTHONPATH=/mnt/c/Auto-research OMP_NUM_THREADS=3 MKL_NUM_THREADS=3
for spec in "door_plan:tests/test_evaluate_door.py" "door_real:tests/test_evaluate_door.py --real"; do
  name=${spec%%:*}; cmd=${spec#*:}
  s=$(date +%s)
  nice -n 15 $PY $cmd > "$O/$name.log" 2>&1; rc=$?
  echo "$name rc=$rc $(( $(date +%s) - s )) s | $(grep -v '^WARNING' "$O/$name.log" | tail -n 1 | cut -c1-200)" >> "$O/summary.txt"
done
