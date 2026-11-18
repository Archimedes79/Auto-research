#!/bin/bash
# E130b / E131b: the four further failure types with perturbed VAL features (fog, radar_off, extr3, cam_front_off): champion single frame and spatial hold
S=/mnt/c/ML-Experiment/scripts
L=/mnt/c/ML-Experiment/out/loop2/e17/logs
P=fog,radar_off,extr3,cam_front_off
[ -f /mnt/c/ML-Experiment/out/loop2/e17/dense_e37/cam_front_off_prob.npy ] || bash $S/_py14.sh p17_dense.py --head e37_final --tag e37 --perts $P > $L/e130b_dense.log 2>&1 || exit 1
bash $S/_py14.sh p17_series.py --dense e37 --perts $P --variants single,maxdecay:0.95 > $L/e130b_series.log 2>&1 || exit 1
bash $S/_py14.sh p17_eval.py --out e130b_more --perts $P --variants single_e37,maxdecay:0.95_e37 > $L/e130b.log 2>&1
echo finished > $L/e130b.done
