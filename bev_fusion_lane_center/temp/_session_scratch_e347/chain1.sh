#!/bin/bash
cd /mnt/c/Auto-research/bev_fusion_lane_center
for t in "jends.py" "grid/test_jmaps_dirx.py" "jend_grid_stats.py selftest" "temp/_session_scratch_e347/stage/training/grid/test_jmaps_r2.py"; do
  echo "=== $t"
  E347_FAST=1 bash temp/e347/run_staged.sh $t 2>&1 | grep -v "^run_staged: stage" | tail -14 | cut -c1-300
done
