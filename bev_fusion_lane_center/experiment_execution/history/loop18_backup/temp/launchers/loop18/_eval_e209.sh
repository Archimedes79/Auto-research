#!/bin/bash
# evaluation part of E209: night and blur series, single frame and spatial hold
R=/mnt/c/ML-Experiment/bev_fusion_lane_center
V=single,maxdecay:0.95
set -e
bash $R/tools/env/_pyr.sh p17_eval.py --out e142 --perts dark,blur --mapnet 1 --ckpt e17_map_b2 --map_mode exact --map_sigma 0 --suffix _B2m0 --variants $V
bash $R/tools/env/_pyr.sh p17_eval.py --out e142 --perts dark,blur --mapnet 1 --ckpt e17_map_b2 --map_mode none --suffix _B2none --variants $V
bash $R/tools/env/_pyr.sh p17_eval.py --out e142 --perts dark,blur --mapnet 1 --ckpt e17_map_c2 --map_mode none --suffix _C2none --variants $V
