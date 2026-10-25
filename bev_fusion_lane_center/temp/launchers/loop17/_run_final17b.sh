#!/bin/bash
S=/mnt/c/ML-Experiment/bev_fusion_lane_center/training
L=/mnt/c/ML-Experiment/bev_fusion_lane_center/runs/e17/logs
bash /mnt/c/ML-Experiment/bev_fusion_lane_center/tools/env/_py14.sh p17_extra.py > $L/final_extra.log 2>&1
bash /mnt/c/ML-Experiment/bev_fusion_lane_center/tools/env/_py14.sh build_loop17_report.py > $L/final_build.log 2>&1
echo finished > $L/final17b.done
