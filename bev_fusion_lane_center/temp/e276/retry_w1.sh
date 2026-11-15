#!/bin/bash
# second retry of E279 seed 1 after the page-allocation failure of 04:32 (memory reclaimed at 04:36); the trainer resumes from ckpt.pt
cd /mnt/c/Auto-research/bev_fusion_lane_center || exit 1
pkill -f "[l]aunch_chain.sh"; sleep 2
rm -f temp/e276/STARTED_w1
bash temp/e276/restart_chain.sh
sleep 3
tail -2 temp/e276/chain.log
