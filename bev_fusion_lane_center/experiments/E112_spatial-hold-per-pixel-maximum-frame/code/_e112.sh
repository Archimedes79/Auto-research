#!/bin/bash
# Loop 12 / E112: spatial hold (per-pixel max with a decayed warped memory), abrupt onset, the four perturbations of E105
O=/mnt/c/ML-Experiment/bev_fusion_lane_center/runs
mkdir -p $O/e112_a
for p in dark blur cam_front3_off radar_off; do
  for v in single; do
    cp -n $O/e105_a/graphs_${p}_${v}.pkl $O/e112_a/ 2>/dev/null
    cp -n $O/e105_a/comps_${p}_${v}.npy $O/e112_a/ 2>/dev/null
  done
done
cd /mnt/c/ML-Experiment/bev_fusion_lane_center/training
bash _pyr.sh p11_temporal.py --perts dark,blur,cam_front3_off,radar_off --variants maxdecay:0.9,maxdecay:0.95 --out e112_a > $O/e112_a/run.log 2>&1
echo done > $O/e112.done
