#!/bin/bash
# wait (max 20 min) until the E315 pair stage of either seed has started (jends_train.py --name e315_p_s*)
for i in $(seq 1 120); do
  if ps aux | grep -E "jends_train.py --name e315_p" | grep -v grep > /dev/null; then echo "e315 pair stage running at $(date +%H:%M:%S)"; ps aux | grep -E "jends_train.py --name e315_p" | grep -v grep | head -2 | cut -c1-120; exit 0; fi
  sleep 10
done
echo "not started within 20 minutes"; tail -n 3 /mnt/c/Auto-research/bev_fusion_lane_center/temp/day11/day11.log
