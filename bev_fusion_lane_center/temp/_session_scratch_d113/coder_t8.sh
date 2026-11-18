#!/bin/bash
cd /mnt/c/Auto-research/bev_fusion_lane_center
export OMP_NUM_THREADS=2
L=temp/_session_scratch_d113
bash tools/env/_py14.sh lane_past_data.py loader-fixture --split val --n-pairs 40 --out $L/coder_lf.json > $L/coder_lf.log 2>&1
bash tools/env/_py14.sh lane_past_data.py fixture --split val --n-pairs 20 --out $L/coder_fx20.json > $L/coder_fx20.log 2>&1
bash tools/env/_py14.sh lane_past_data.py potential --split val --seed 0 --n 60 --out $L/coder_pot60.json > $L/coder_pot60.log 2>&1
echo finished > $L/coder_t8.done
