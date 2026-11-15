#!/bin/bash
cd /mnt/c/Auto-research/bev_fusion || exit 1
export AR_PROJECT=../bev_fusion
date
echo "--- waiter"
ps -eo pid,etime,args | grep -E '[n]ight_B2|[w]indow_gate' | cut -c1-150
echo "--- schedule (t23 / t22 rows)"
python3 ../autoresearch/tools/schedule.py show 2>&1 | cut -c1-200 | head -8
