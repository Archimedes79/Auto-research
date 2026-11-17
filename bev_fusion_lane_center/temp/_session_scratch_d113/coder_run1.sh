#!/bin/bash
cd /mnt/c/Auto-research/bev_fusion_lane_center
export OMP_NUM_THREADS=2
bash tools/env/_py14.sh lane_past_data.py build --all
bash tools/env/_py14.sh lane_past_data.py fixture --split val --n-pairs 200
