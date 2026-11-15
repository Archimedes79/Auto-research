#!/bin/bash
# scratch: exits when the two B2 bootstrap comparisons have ended (one notification), and prints the state of the L4 check
cd /mnt/c/Auto-research/bev_fusion || exit 1
while pgrep -f "bev_suite.py compare runs/t2[12]_.*runs/t22_B2_s0" > /dev/null; do sleep 20; done
echo "B2 comparisons ended $(date +%H:%M:%S)"
tail -n 3 temp/t22/compare_B2_vs_control.log | cut -c1-200
tail -n 3 temp/t22/compare_B2_vs_B.log | cut -c1-200
echo "--- L4 check state"
ls records/runs | grep diag.t20 || echo "no diag.t20 run record yet"
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
