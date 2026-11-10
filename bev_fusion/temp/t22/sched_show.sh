#!/bin/bash
# scratch: the schedule of bev_fusion (the t22 rows)
cd /mnt/c/Auto-research/autoresearch || exit 1
export AR_PROJECT=../bev_fusion
python3 tools/schedule.py show 2>&1 | cut -c1-260 | head -40
echo ---
grep -n 't22\|frozen.update' /mnt/c/Auto-research/bev_fusion/records/schedule.csv | cut -c1-400
