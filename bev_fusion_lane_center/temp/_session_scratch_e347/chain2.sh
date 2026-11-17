#!/bin/bash
cd /mnt/c/Auto-research/bev_fusion_lane_center
O=temp/_session_scratch_e347
echo "=== real smoke (r2 test --real, no unit tests)"
E347_FAST=1 bash temp/e347/run_staged.sh $O/stage/training/grid/test_jmaps_r2.py --real --only none 2>&1 | grep -v "^run_staged: stage" | tail -12 | cut -c1-400
echo "=== GT statistics on real VAL"
E347_FAST=1 bash temp/e347/run_staged.sh jend_grid_stats.py --out $O/jend_grid_stats_val.json 2>&1 | grep -v "^run_staged: stage" | tail -12 | cut -c1-700
echo "=== scorer old vs new"
bash temp/e347/score_old_new.sh e332_ma_s0 24 2>&1 | tail -8
echo "=== mutations"
python3 temp/e347/mutate.py 2>&1 | tail -40
echo "=== chain2 done"
