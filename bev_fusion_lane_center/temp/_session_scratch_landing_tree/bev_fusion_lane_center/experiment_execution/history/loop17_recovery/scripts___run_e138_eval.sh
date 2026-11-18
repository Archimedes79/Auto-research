#!/bin/bash
# E138: map-conditioned builder on the failure series: exact SD map (pose error 0 and 2 m), no map, wrong map; single frame and spatial hold; plus the clean half is part of every series
S=/mnt/c/ML-Experiment/scripts
L=/mnt/c/ML-Experiment/out/loop2/e17/logs
V=single,maxdecay:0.95
bash $S/_py14.sh p17_eval.py --out e138_map --mapnet 1 --ckpt e17_map_a --map_mode exact --map_sigma 0 --suffix _m0 --variants $V > $L/e138_m0.log 2>&1
bash $S/_py14.sh p17_eval.py --out e138_map --mapnet 1 --ckpt e17_map_a --map_mode exact --map_sigma 2 --suffix _m2 --variants $V > $L/e138_m2.log 2>&1
bash $S/_py14.sh p17_eval.py --out e138_map --mapnet 1 --ckpt e17_map_a --map_mode none --suffix _mnone --variants $V > $L/e138_mnone.log 2>&1
bash $S/_py14.sh p17_eval.py --out e138_map --mapnet 1 --ckpt e17_map_a --map_mode wrong --suffix _mwrong --variants $V > $L/e138_mwrong.log 2>&1
echo finished > $L/e138_eval.done
