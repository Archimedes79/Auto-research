#!/bin/bash
cd /mnt/c/Auto-research/bev_fusion_lane_center
export E347_FAST=1
echo "=== jend_grid_stats selftest"
bash temp/e347/run_staged.sh jend_grid_stats.py selftest 2>&1 | grep -v "^run_staged: stage" | tail -3
echo "=== mutations m28 m29"
python3 temp/e347/mutate.py m28_dropped_counts_members m29_adjacent_cells_wide 2>&1 | cut -c1-300
echo "=== jmaps_eval selftest"
bash temp/e347/run_staged.sh jmaps_eval.py selftest 2>&1 | grep -v "^run_staged: stage" | tail -2
echo "=== test_jmaps_r2 full + real"
bash temp/e347/run_staged.sh temp/_session_scratch_e347/stage/training/grid/test_jmaps_r2.py --real 2>&1 | grep -v "^run_staged: stage" | cut -c1-330 | tail -12
echo "=== chain6 done"
