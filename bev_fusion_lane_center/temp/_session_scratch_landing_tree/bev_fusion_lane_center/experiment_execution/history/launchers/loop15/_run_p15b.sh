#!/bin/bash
O=/mnt/c/ML-Experiment/bev_fusion_lane_center/runs
rm -f $O/e120b.done
mkdir -p $O/e120_b
bash /mnt/c/ML-Experiment/bev_fusion_lane_center/tools/env/_py14.sh p15_refine.py --name e120_b --epochs 8 --bs 8 --workers 3 --detach 1 > $O/e120_b/run.log 2>&1
echo done > $O/e120b.done
