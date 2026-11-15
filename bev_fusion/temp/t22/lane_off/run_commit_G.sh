#!/bin/bash
# scratch: add the schedule row, check that schedule.py reads it, then commit G
cd /mnt/c/Auto-research/bev_fusion || exit 1
export AR_PROJECT=../bev_fusion
python3 temp/t22/lane_off/sched_add.py || exit 1
python3 ../autoresearch/tools/schedule.py show 2>&1 | cut -c1-220 | head -6
bash temp/t22/lane_off/commit_G.sh
