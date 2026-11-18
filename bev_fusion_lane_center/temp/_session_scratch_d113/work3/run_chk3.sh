#!/bin/bash
cd /mnt/c/Auto-research/bev_fusion_lane_center
export OMP_NUM_THREADS=1
W=temp/_session_scratch_d113/work3
bash tools/env/_py14.sh lane_map_data.py check --n 60 --out $W/chk60.json > $W/chk60.txt 2>&1
echo "exit $?" >> $W/chk60.txt
bash tools/env/_py14.sh lane_map_data.py check --out $W/chk_full704.json > $W/chk_full704.txt 2>&1
echo "exit $?" >> $W/chk_full704.txt
