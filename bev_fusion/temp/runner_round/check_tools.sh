#!/bin/bash
cd /mnt/c/Auto-research/bev_fusion || exit 1
export AR_PROJECT=/mnt/c/Auto-research/bev_fusion
echo "--- schedule show (BEV) ---"
python3 ../autoresearch/tools/schedule.py show 2>&1 | cut -c1-120 | head -14
echo "--- plan status (BEV) ---"
python3 ../autoresearch/tools/plan.py status 2>&1 | cut -c1-120 | head -8
echo "--- D67 / D68 rows in HEAD ---"
git show HEAD:bev_fusion_lane_center/experiment_execution/DECISIONS.md | grep -c -E '^\| 2026-11-09 \| D6[78] '
