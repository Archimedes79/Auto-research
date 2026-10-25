#!/bin/bash
# R2.1 / E213: map-conditioned builder B (e17_map_b, E139) on the VAL failure series with the SD map after rough localisation (K145) and trust gate (K146).
# registration output: runs/e18/regis_eval_a (p18_regis_eval.py, regis_a); builder results: runs/e17/e18_r21/<pert>_maxdecay_0.95<suffix>.pkl
# baselines without registration are in runs/e17/e139_b (_m0 exact, _m2 sigma 2, _mnone, _mwrong)
R=/mnt/c/ML-Experiment/bev_fusion_lane_center
RG=$R/runs/e18/regis_eval_a
P=dark,blur,cam_front3_off
V=maxdecay:0.95
E="bash $R/tools/env/_pyr.sh p17_eval.py --out e18_r21 --perts $P --mapnet 1 --ckpt e17_map_b --variants $V"
set -e
for s in 2 1 3; do
  $E --map_mode exact --map_sigma $s --reg $RG --reg_trust hard --reg_thr 0.62 --suffix _R62_s$s
done
$E --map_mode wrong --reg $RG --reg_trust hard --reg_thr 0.62 --suffix _R62_wrong
$E --map_mode wrong --reg $RG --reg_trust hard --reg_thr 0.75 --suffix _R75_wrong
$E --map_mode exact --map_sigma 2 --reg $RG --reg_trust hard --reg_thr 0.75 --suffix _R75_s2
$E --map_mode exact --map_sigma 2 --reg $RG --reg_trust none --suffix _Rnotrust_s2
$E --map_mode exact --map_sigma 1 --suffix _U_s1
$E --map_mode exact --map_sigma 3 --suffix _U_s3
