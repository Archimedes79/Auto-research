#!/bin/bash
# final numbers of the report: master, extra, figures, report html
S=/mnt/c/ML-Experiment/bev_fusion_lane_center/training
L=/mnt/c/ML-Experiment/bev_fusion_lane_center/runs/e17/logs
bash /mnt/c/ML-Experiment/bev_fusion_lane_center/tools/env/_py14.sh p17_master.py > $L/final_master.log 2>&1
bash /mnt/c/ML-Experiment/bev_fusion_lane_center/tools/env/_py14.sh p17_extra.py > $L/final_extra.log 2>&1
bash /mnt/c/ML-Experiment/bev_fusion_lane_center/tools/env/_py14.sh fig_p17_bars.py > $L/final_figs.log 2>&1
bash /mnt/c/ML-Experiment/bev_fusion_lane_center/tools/env/_py14.sh fig_p17_best.py >> $L/final_figs.log 2>&1
bash /mnt/c/ML-Experiment/bev_fusion_lane_center/tools/env/_py14.sh build_loop17_report.py > $L/final_build.log 2>&1
echo finished > $L/final17.done
