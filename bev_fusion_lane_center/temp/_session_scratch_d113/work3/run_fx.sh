#!/bin/bash
cd /mnt/c/Auto-research/bev_fusion_lane_center
export OMP_NUM_THREADS=2
W=temp/_session_scratch_d113/work3
bash tools/env/_py14.sh lane_map_data.py fixture --split val --n 100 --out $W/fx100.json > $W/fx100.txt 2>&1
bash tools/env/_py14.sh lane_map_data.py fixture --split val --n 20 --out $W/fx20.json > $W/fx20.txt 2>&1
bash tools/env/_py14.sh lane_map_data.py potential --split val --seed 0 --n 60 --out $W/pot60.json > $W/pot60.txt 2>&1
bash tools/env/_py14.sh lane_map_data.py check --n 60 --out $W/chk60.json > $W/chk60.txt 2>&1
echo done > $W/fx.done
