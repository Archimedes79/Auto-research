#!/bin/bash
# E291: queue the pair stages (arms given as "c 0" "b 0" ...) into run.py's FIFO GPU queue, each in its own process; logs in temp/_session_scratch_s292/logs
cd /mnt/c/Auto-research/bev_fusion_lane_center || exit 1
for X in "$@"; do
  set -- $X
  bash temp/e291/start_arms.sh $1 $2 > temp/_session_scratch_s292/logs/e291_$1_s$2.log 2>&1 &
  sleep 20
done
wait
echo "e291 launcher done $(date +%H:%M:%S)"
