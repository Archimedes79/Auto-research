#!/bin/bash
# Loop 12 / E109: calibration of the probabilities handed to the planner (A: CPU, B: GPU)
O=/mnt/c/ML-Experiment/bev_fusion_lane_center/runs
cd /mnt/c/ML-Experiment/bev_fusion_lane_center/training
bash _py.sh p12_calib.py > $O/e109_calib_a.log 2>&1
bash _pyr.sh p12_calib_b.py > $O/e109_calib_b.log 2>&1
echo done > $O/e109.done
