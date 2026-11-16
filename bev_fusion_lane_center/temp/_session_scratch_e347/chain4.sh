#!/bin/bash
cd /mnt/c/Auto-research/bev_fusion_lane_center
export E347_FAST=1
echo "=== strengthened tests on the unmutated stage"
bash temp/e347/run_staged.sh temp/_session_scratch_e347/stage/training/grid/test_jmaps_r2.py --only loss,keypoints,pair_head 2>&1 | grep -v "^run_staged" | cut -c1-250
echo "=== the three survivors again"
python3 temp/e347/mutate.py m07_offset_unmasked m08_off_weight m15_softargmax_res 2>&1 | cut -c1-400
echo "=== chain4 done"
