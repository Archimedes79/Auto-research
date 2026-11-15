#!/bin/bash
cd /mnt/c/Auto-research/bev_fusion_lane_center
export E347_FAST=1
O=temp/_session_scratch_e347
echo "=== jend_grid_stats selftest"
bash temp/e347/run_staged.sh jend_grid_stats.py selftest 2>&1 | grep -v "^run_staged: stage" | tail -3
echo "=== jend_grid_stats on VAL"
bash temp/e347/run_staged.sh jend_grid_stats.py --out $O/jend_grid_stats_val.json 2>&1 | grep -v "^run_staged" | cut -c1-700
echo "=== mutations of the stats tool and of the wiring"
python3 temp/e347/mutate.py m28_dropped_counts_members m29_adjacent_cells_wide m25_data_r2_not_passed m26_jmaps_train_no_r2_data m27_jends_train_no_r2_data 2>&1 | cut -c1-330
echo "=== chain5 done"
