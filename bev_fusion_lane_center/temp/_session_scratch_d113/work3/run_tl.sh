#!/bin/bash
cd /mnt/c/Auto-research/bev_fusion_lane_center
export OMP_NUM_THREADS=1
W=temp/_session_scratch_d113/work3
bash tools/env/_py14.sh test_lane_map.py > $W/tl_out.txt 2>&1
echo "exit $?" >> $W/tl_out.txt
