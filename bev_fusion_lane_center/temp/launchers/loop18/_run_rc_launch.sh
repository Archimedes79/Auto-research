#!/bin/bash
L=/mnt/c/ML-Experiment/bev_fusion_lane_center/temp/launchers/loop18
bash $L/_run_rc.sh rc2 6.0 &
sleep 90
bash $L/_run_rc.sh rc3 10.0 &
wait
