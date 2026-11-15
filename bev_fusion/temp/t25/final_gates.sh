#!/bin/bash
cd /mnt/c/Auto-research/bev_fusion || exit 1
export PYTHONPATH=/mnt/c/Auto-research OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
PY=~/miniconda3/envs/gcar/bin/python
for t in tests/test_radar_recursive.py training/precompute.py training/interface.py training/lift_pd.py; do
  s=$(date +%s); nice -n 15 $PY $t > /tmp/t25_gate.log 2>&1; rc=$?
  echo "$t rc=$rc $(( $(date +%s) - s )) s | $(grep -v '^WARNING' /tmp/t25_gate.log | tail -n 1 | cut -c1-160)"
done
