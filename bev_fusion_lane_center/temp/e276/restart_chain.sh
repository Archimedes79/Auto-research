#!/bin/bash
# scratch: restart the E276 scheduler (only when no job of it runs: it keeps no state besides the STARTED_* / done flags)
cd /mnt/c/Auto-research/bev_fusion_lane_center || exit 1
pkill -f "[l]aunch_chain.sh"
sleep 2
nohup bash temp/e276/launch_chain.sh > temp/e276/chain.out 2>&1 < /dev/null &
sleep 3
pgrep -af "[l]aunch_chain" | head -2 | cut -c1-80
head -3 temp/e276/chain.log
