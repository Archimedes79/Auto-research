#!/bin/bash
cd /mnt/c/Auto-research/bev_fusion_lane_center
export OMP_NUM_THREADS=1
W=temp/_session_scratch_d113/work3
bash tools/env/_py14.sh lane_map_data.py fixture --split val --n 100 --out $W/fx100b.json > $W/fx100b.txt 2>&1
echo "exit $?" >> $W/fx100b.txt
bash tools/env/_py14.sh lane_map_data.py fixture --split val --n 20 --out $W/fx20.json > $W/fx20.txt 2>&1
echo "exit $?" >> $W/fx20.txt
bash tools/env/_py14.sh lane_map_data.py potential --split val --seed 0 --n 60 --out $W/pot60.json > $W/pot60.txt 2>&1
echo "exit $?" >> $W/pot60.txt
bash tools/env/_py14.sh lane_map_data.py check --n 60 --out $W/chk60.json > $W/chk60.txt 2>&1
echo "exit $?" >> $W/chk60.txt
bash tools/env/_py14.sh lane_map_data.py potential --split val --seed 0 --out $W/pot_cut_full.json > $W/pot_cut_full.txt 2>&1
echo "exit $?" >> $W/pot_cut_full.txt
bash tools/env/_py14.sh lane_map_data.py check --out $W/chk_cut_full.json > $W/chk_cut_full.txt 2>&1
echo "exit $?" >> $W/chk_cut_full.txt
