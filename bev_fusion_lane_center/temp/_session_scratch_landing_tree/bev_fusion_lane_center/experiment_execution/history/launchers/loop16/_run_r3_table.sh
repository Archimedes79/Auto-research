#!/bin/bash
# rung 3 table: every rung-3 checkpoint at the common assembly (cov_r 0.5, vtx_r 1.5; junction anchoring 0 and 0.5), beam 192 / 96 and 288 / 144, VAL250
S=/mnt/c/ML-Experiment/bev_fusion_lane_center/training
for N in c1 t1 j1 f1 g0; do
  for MK in "192 96" "288 144"; do
    set -- $MK
    bash /mnt/c/ML-Experiment/bev_fusion_lane_center/tools/env/_py14.sh p16_asm_grid.py --design D7 --name e123_D7_r3_$N --n 250 --cov 0.5 --vtx 1.5 --junc 0 0.5 --tag r3tab_M$1 --set M=$1 K0=$2 > /mnt/c/ML-Experiment/bev_fusion_lane_center/runs/e123_ladder/r3tab_${N}_M$1.log 2>&1
  done
done
echo done > /mnt/c/ML-Experiment/bev_fusion_lane_center/runs/e123_ladder/r3tab.done
