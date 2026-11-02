#!/bin/bash
# the CPU tests of the speed work and the scaler, from bev_fusion/ (nice: a training may run on the box)
cd /mnt/c/Auto-research/bev_fusion || exit 1
export PYTHONPATH="$(cd .. && pwd)"
for t in test_host_overhead test_speed_flags test_grad_scaler test_gt_cache_targets; do
  printf "%-26s" $t
  OMP_NUM_THREADS=2 nice -n 19 ~/miniconda3/envs/gcar/bin/python tests/$t.py > /tmp/rt_$t.log 2>&1; rc=$?
  echo "exit $rc | $(grep -E 'GATE|all checks pass' /tmp/rt_$t.log | tail -1)"
done
