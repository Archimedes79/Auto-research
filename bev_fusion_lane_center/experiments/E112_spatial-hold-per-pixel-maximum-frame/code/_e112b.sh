#!/bin/bash
# Loop 12 / E112b: the spatial hold under a gradual onset (8-frame ramp), night and blur; waits for E108b and E112
O=/mnt/c/ML-Experiment/bev_fusion_lane_center/runs
while [ ! -f $O/e108b.done ] || [ ! -f $O/e112.done ]; do sleep 20; done
mkdir -p $O/e112_ramp8
cd /mnt/c/ML-Experiment/bev_fusion_lane_center/training
bash _pyr.sh p11_temporal.py --perts dark,blur --variants maxdecay:0.95 --out e112_ramp8 --ramp 8 > $O/e112_ramp8/run.log 2>&1
echo done > $O/e112b.done
