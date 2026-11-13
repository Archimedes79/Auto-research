#!/bin/bash
# retry E279 after the host-memory failure of 04:18 (OSError ENOMEM in the DataLoader workers of both jobs): the trainer resumes from the run's ckpt.pt
cd /mnt/c/Auto-research/bev_fusion_lane_center || exit 1
pkill -f "[l]aunch_chain.sh"; sleep 2
rm -f temp/e276/STARTED_w0 temp/e276/STARTED_w1
free -m | head -2
ls runs/e279_w_s0_p3 runs/e279_w_s1_p3 | head -20
bash temp/e276/restart_chain.sh
sleep 3
tail -2 temp/e276/chain.log
