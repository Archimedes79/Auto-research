#!/bin/bash
cd /mnt/c/Auto-research/bev_fusion_lane_center
L=temp/e358/all_checks3.log
: > $L
bash temp/e358/run_scratch.sh temp/_session_scratch_e358/test_staged.py >> $L 2>&1
bash temp/e358/run_scratch.sh temp/_session_scratch_e358/mutate.py jwire_head > temp/e358/mut_jwire_head.log 2>&1
echo "== mutate jwire_head: $(tail -1 temp/e358/mut_jwire_head.log)" >> $L
grep "SURVIVOR\|NOT APPLIED" temp/e358/mut_jwire_head.log >> $L
bash temp/e358/make_patch.sh >> $L 2>&1
echo ALL DONE >> $L
