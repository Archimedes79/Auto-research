#!/bin/bash
# smoke test of p18_joint.py: 24 steps with failure rows, checkpoint every 10, then a second start that must resume from 20
R=/mnt/c/ML-Experiment/bev_fusion_lane_center
N=e18_smoke_joint
rm -rf $R/runs/$N $R/runs/e123/$N
A="--name $N --iters 24 --bs 4 --lam 0.05 --real_fail 1 --share 0.5 --ckpt_every 10 --workers 3"
bash $R/tools/env/_py14.sh p18_joint.py $A 2>&1 | tail -6
echo "=== second start (must resume from 20) ==="
rm -f $R/runs/$N/model.pt $R/runs/$N/train_summary.json
bash $R/tools/env/_py14.sh p18_joint.py $A 2>&1 | tail -6
ls -la $R/runs/$N
