#!/bin/bash
R=/mnt/c/ML-Experiment/bev_fusion_lane_center
bash $R/tools/env/_py14.sh p18_final_boot.py > $R/runs/e18/final_boot.log 2> $R/runs/e18/final_boot.err
cat $R/runs/e18/final_boot.log
