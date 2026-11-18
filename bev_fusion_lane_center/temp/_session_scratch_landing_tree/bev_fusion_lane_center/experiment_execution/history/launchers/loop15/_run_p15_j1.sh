#!/bin/bash
# joint arms with the position loss, one epoch each (the two-epoch runs were too slow on the shared GPU)
O=/mnt/c/ML-Experiment/bev_fusion_lane_center/runs
S=/mnt/c/ML-Experiment/bev_fusion_lane_center/training
rm -f $O/e121b_j1.done $O/e121_b/result_joint1.json $O/e121_b_keep/result_joint1.json
(bash /mnt/c/ML-Experiment/bev_fusion_lane_center/tools/env/_py14.sh p15_joint.py --arm joint1 --out e121_b --epochs 1 --workers 2 > $O/e121_b/joint1.log 2>&1) &
bash /mnt/c/ML-Experiment/bev_fusion_lane_center/tools/env/_py14.sh p15_joint.py --arm joint1 --keep_in 1 --out e121_b_keep --epochs 1 --workers 2 > $O/e121_b_keep/joint1_keep.log 2>&1
wait
echo done > $O/e121b_j1.done
