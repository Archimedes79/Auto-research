#!/bin/bash
# the four real-data gates on a run made after D76 (D81), one after the other, low priority; results in /tmp/real_gates_mine/
cd /mnt/c/Auto-research/bev_fusion || exit 1
O=/tmp/real_gates_mine; rm -rf "$O"; mkdir -p "$O"
PY=~/miniconda3/envs/gcar/bin/python
export PYTHONPATH=/mnt/c/Auto-research OMP_NUM_THREADS=3 MKL_NUM_THREADS=3
bash -n tests/run_gates.sh && echo "run_gates.sh syntax ok" > "$O/summary.txt"
for spec in "bev_free:tests/test_bev_free.py" "evaluation_api_real:tests/test_evaluation_api.py --real" "evaluate_door_real:tests/test_evaluate_door.py --real" "runner_real:tests/test_runner_real.py"; do
  name=${spec%%:*}; cmd=${spec#*:}
  s=$(date +%s)
  nice -n 15 $PY $cmd > "$O/$name.log" 2>&1; rc=$?
  echo "$name rc=$rc $(( $(date +%s) - s )) s | $(tail -n 1 "$O/$name.log" | cut -c1-170)" >> "$O/summary.txt"
done
echo "done" >> "$O/summary.txt"
