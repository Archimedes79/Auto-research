#!/bin/bash
# E140 with a map pose error of 1 m (the builder sees the map shifted by a scene-constant 1 m error)
S=/mnt/c/ML-Experiment/recovered_loop17/tree/scripts
L=/mnt/c/ML-Experiment/recovered_loop17/tree/out/loop2/e17/logs
bash $S/_py14.sh p17_eval.py --out e140_gateB --mapnet 1 --ckpt e17_map_b --map_mode exact --map_sigma 1 --suffix _m1 --variants gateB > $L/e140_eval_m1.log 2>&1
echo finished > $L/e140_m1.done
