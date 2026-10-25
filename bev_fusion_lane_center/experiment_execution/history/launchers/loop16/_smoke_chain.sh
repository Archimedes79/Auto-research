#!/bin/bash
# plumbing test of the builder -> route head -> chain data -> consumer path on small subsets (D7 r1 b1)
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate gcar
source /mnt/c/ML-Experiment/bev_fusion_lane_center/tools/env/bflc_env.sh
export PYTHONWARNINGS=ignore PYTHONUNBUFFERED=1
cd /mnt/c/ML-Experiment/bev_fusion_lane_center/training
python p16_tokens.py --design D7 --name e123_D7_r1_b1 --out tok_p16_smoke --split train --n 240 2>&1 | tail -1
python p16_tokens.py --design D7 --name e123_D7_r1_b1 --out tok_p16_smoke --split val_clean --n 80 2>&1 | tail -1
python p16_tokens.py --design D7 --name e123_D7_r1_b1 --out tok_p16_smoke --split test_clean --n 80 2>&1 | tail -1
python p10_route_train.py --tok tok_p16_smoke_train --name p16_route_smoke --iters 150 --xattn --workers 2 2>&1 | tail -2
python p10_route_eval2.py --tok tok_p16_smoke_test_clean --split test_clean --model p16_route_smoke --kind x --cells exact,none --tag smoke 2>&1 | tail -4
