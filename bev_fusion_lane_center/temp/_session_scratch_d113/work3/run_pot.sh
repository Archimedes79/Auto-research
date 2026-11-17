#!/bin/bash
cd /mnt/c/Auto-research/bev_fusion_lane_center
export OMP_NUM_THREADS=2
bash tools/env/_py14.sh temp/_session_scratch_d113/step0_map2.py > temp/_session_scratch_d113/work3/step0_out.txt 2>&1
bash tools/env/_py14.sh lane_map_data.py potential --split val --seed 0 --out temp/_session_scratch_d113/work3/pot_full.json > temp/_session_scratch_d113/work3/pot_out.txt 2>&1
echo done > temp/_session_scratch_d113/work3/pot.done
