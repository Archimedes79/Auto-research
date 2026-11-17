#!/bin/bash
cd /mnt/c/Auto-research/bev_fusion_lane_center
export OMP_NUM_THREADS=1
W=temp/_session_scratch_d113/work3
bash tools/env/_py14.sh temp/_session_scratch_d113/work3/step0_full.py > $W/step0_full.txt 2>&1
echo done > $W/s0.done
