#!/bin/bash
cd /mnt/c/Auto-research/bev_fusion || exit 1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=""
PY=$HOME/miniconda3/envs/gcar/bin/python
for t in test_student_encoder test_slots test_host_batch test_speed_flags test_lane_branch_chain; do
  s=$(date +%s)
  $PY tests/$t.py > temp/t22/other_$t.log 2>&1
  echo "$t exit $? ($(( $(date +%s) - s )) s): $(tail -1 temp/t22/other_$t.log | cut -c1-120)"
done
