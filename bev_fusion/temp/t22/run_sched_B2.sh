#!/bin/bash
cd /mnt/c/Auto-research/bev_fusion || exit 1
export AR_PROJECT=../bev_fusion
python3 temp/t22/sched_B2.py
git -C /mnt/c/Auto-research diff --stat -- bev_fusion/records/schedule.csv | tail -2
python3 ../autoresearch/tools/schedule.py show 2>&1 | cut -c1-200 | head -12
