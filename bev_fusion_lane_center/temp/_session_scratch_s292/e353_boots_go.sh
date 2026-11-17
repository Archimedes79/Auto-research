#!/bin/bash
cd /mnt/c/Auto-research/bev_fusion_lane_center || exit 1
rm -f temp/e353/boots.log temp/e353/boot_*.log
for s in 0 1; do
  nohup bash temp/_session_scratch_s292/e353_boots.sh $s > /dev/null 2>&1 &
done
sleep 2
ps -eo pid,etime,cmd | grep e353_boots | grep -v grep | cut -c1-90
