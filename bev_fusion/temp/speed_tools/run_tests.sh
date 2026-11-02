#!/bin/bash
# the two CPU tests of the speed work, from bev_fusion/
cd /mnt/c/Auto-research/bev_fusion || exit 1
for t in test_host_overhead test_speed_flags; do
  echo "=== $t"
  OMP_NUM_THREADS=2 ~/miniconda3/envs/gcar/bin/python tests/$t.py 2>&1 | tail -${1:-14} | cut -c1-230
done
