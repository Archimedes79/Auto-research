#!/bin/bash
S=/mnt/c/ML-Experiment/bev_fusion_lane_center/training
L=/mnt/c/ML-Experiment/bev_fusion_lane_center/runs/e17/logs
until [ -f $L/e140_route.done ]; do sleep 15; done
bash /mnt/c/ML-Experiment/bev_fusion_lane_center/tools/env/_py14.sh p17_master.py > $L/master3.log 2>&1
bash /mnt/c/ML-Experiment/bev_fusion_lane_center/tools/env/_py14.sh fig_p17_bars.py >> $L/master3.log 2>&1
bash /mnt/c/ML-Experiment/bev_fusion_lane_center/tools/env/_py14.sh fig_p17_best.py >> $L/master3.log 2>&1
echo finished > $L/master3.done
