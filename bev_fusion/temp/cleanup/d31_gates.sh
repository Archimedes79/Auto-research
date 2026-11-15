#!/bin/bash
# D31 evaluation change: fast gates, then the real door golden check and the API real gate on chain2_t20; results in /tmp/d31_gates_mine/
cd /mnt/c/Auto-research/bev_fusion || exit 1
O=/tmp/d31_gates_mine; rm -rf "$O"; mkdir -p "$O"
PY=~/miniconda3/envs/gcar/bin/python
export PYTHONPATH=/mnt/c/Auto-research OMP_NUM_THREADS=3 MKL_NUM_THREADS=3
s=$(date +%s); nice -n 15 bash tests/run_gates.sh > "$O/fast.log" 2>&1; rc=$?
echo "fast gates rc=$rc $(( $(date +%s) - s )) s | $(tail -n 2 "$O/fast.log" | tr '\n' ' ' | cut -c1-200)" >> "$O/summary.txt"
for spec in "door_real:tests/test_evaluate_door.py --real" "api_real:tests/test_evaluation_api.py --real"; do
  name=${spec%%:*}; cmd=${spec#*:}
  s=$(date +%s)
  nice -n 15 $PY $cmd > "$O/$name.log" 2>&1; rc=$?
  echo "$name rc=$rc $(( $(date +%s) - s )) s | $(grep -v '^WARNING' "$O/$name.log" | tail -n 1 | cut -c1-200)" >> "$O/summary.txt"
done
echo done >> "$O/summary.txt"
