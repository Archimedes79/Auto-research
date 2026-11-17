#!/bin/bash
# E347 stage 1: the four arm runs go into run.py's FIFO GPU queue in this order (2 slots): g_s0, o_s0, g_s1, o_s1
cd /mnt/c/Auto-research/bev_fusion_lane_center || exit 1
mkdir -p temp/_session_scratch_s292/logs
for X in "g 0" "o 0" "g 1" "o 1"; do
  set -- $X
  bash temp/e347/start_arm.sh $1 $2 > temp/_session_scratch_s292/logs/e347_$1_s$2.log 2>&1 &
  sleep 20
done
wait
echo "e347 stage 1 launcher done $(date +%H:%M:%S)"
