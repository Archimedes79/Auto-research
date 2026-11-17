#!/bin/bash
cd /mnt/c/Auto-research/bev_fusion_lane_center
export OMP_NUM_THREADS=1
W=temp/_session_scratch_d113/work3
bash tools/env/_py14.sh lane_map_data.py check --n 60 --out $W/chk60.json > $W/chk60.txt 2>&1
bash tools/env/_py14.sh lane_map_data.py check --out $W/chk_full.json > $W/chk_full.txt 2>&1
echo done > $W/chk_full.done
