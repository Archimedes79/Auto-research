#!/bin/bash
O=/mnt/c/ML-Experiment/bev_fusion_lane_center/runs
rm -f $O/e120d.done
mkdir -p $O/e120_d
bash /mnt/c/ML-Experiment/bev_fusion_lane_center/tools/env/_py14.sh p15_refine.py --name e120_d --epochs 6 --bs 8 --workers 3 --du_max 0 > $O/e120_d/run.log 2>&1
echo done > $O/e120d.done
