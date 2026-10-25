#!/bin/bash
# smoke test of the resume patch of p17_mapft.py: 24 steps with a checkpoint every 10, then a second start that must resume from step 20
R=/mnt/c/ML-Experiment/bev_fusion_lane_center
N=e18_smoke_mapft
rm -rf $R/runs/e123/$N
FP=cam_front3_off,fog,radar_off,extr3,cam_front_off
bash $R/tools/env/_py14.sh p17_mapft.py --name $N --init_ckpt e17_map_a --real_fail 1 --share 0.5 --u 0.02 --lr 3e-4 --p_degrade 0.2 --fail_perts $FP --ckpt_every 10 --workers 3 2>&1 | tail -15
echo "=== second start (must resume from 20) ==="
rm -f $R/runs/e123/$N/model.pt $R/runs/e123/$N/status.json
bash $R/tools/env/_py14.sh p17_mapft.py --name $N --init_ckpt e17_map_a --real_fail 1 --share 0.5 --u 0.02 --lr 3e-4 --p_degrade 0.2 --fail_perts $FP --ckpt_every 10 --workers 3 2>&1 | tail -15
ls -la $R/runs/e123/$N
