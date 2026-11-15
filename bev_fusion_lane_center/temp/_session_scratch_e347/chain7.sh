#!/bin/bash
cd /mnt/c/Auto-research/bev_fusion_lane_center
export E347_FAST=1
O=temp/_session_scratch_e347
echo "=== NEW tests against the ROUND-1 code (stage_r1): the sections of the round-2 rules must fail"
BFLC_STAGE=/mnt/c/Auto-research/bev_fusion_lane_center/$O/stage_r1 bash temp/e347/run_staged.sh $O/control/run_each.py 2>&1 | grep -E "RESULT"
echo "=== chain7 done"
