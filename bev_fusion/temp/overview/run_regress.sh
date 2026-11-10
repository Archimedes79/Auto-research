#!/bin/bash
cd /mnt/c/Auto-research/bev_fusion || exit 1
export OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=
PY=$HOME/miniconda3/envs/gcar/bin/python
for t in tests/test_lane_branch_dual.py tests/test_frozen.py tests/test_slots.py; do
  echo "=== $t"
  nice -n 12 $PY $t 2>&1 | tail -n 6
  echo "exit ${PIPESTATUS[0]}"
done
