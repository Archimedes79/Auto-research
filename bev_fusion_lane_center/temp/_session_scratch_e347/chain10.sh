#!/bin/bash
cd /mnt/c/Auto-research/bev_fusion_lane_center
O=temp/_session_scratch_e347
export E347_FAST=1
run() { echo "=== $*"; bash temp/e347/run_staged.sh "$@" 2>&1 | grep -v "^run_staged: stage" | tail -${TAILN:-6} | cut -c1-330; echo "exit ${PIPESTATUS[0]}"; }
TAILN=3 run grid/test_jmaps_r2.py
TAILN=1 run jgt.py
TAILN=1 run qjunction.py
TAILN=1 run jmaps_eval.py selftest
TAILN=3 run grid/test_jmaps_r2.py --real --only none
echo "=== mutation m01 (one-hot of an off-map end) + m42"
python3 temp/e347/mutate.py m01_offgrid_end_clipped_into_map m42_names_without_joffm > $O/mut_round2b.log 2>&1
cat $O/mut_round2b.log | cut -c1-230
echo "=== chain10 done"
