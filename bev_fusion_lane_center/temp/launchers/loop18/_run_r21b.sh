#!/bin/bash
# R2.1b / E213.b: the same registration + trust gate in front of the best chain of loop 17 (map builder B + learned gate, E140 "gateB" series); baselines in runs/e17/e140_gateB (_m0 exact, _m1, _m2, _mnone, _mwrong)
R=/mnt/c/ML-Experiment/bev_fusion_lane_center
X=$R/experiment_execution/runs
until [ -f $X/E213.a/done ]; do sleep 30; done
RG=$R/runs/e18/regis_eval_a
T=dark_gateB,blur_gateB,cam_front3_off_gateB
set -e
bash $R/tools/env/_pyr.sh p18_regis_eval.py --regis regis_a --out regis_eval_a --tags $T --sigmas 1,2,3
E="bash $R/tools/env/_pyr.sh p17_eval.py --out e18_r21b --perts dark,blur,cam_front3_off --mapnet 1 --ckpt e17_map_b --variants gateB"
for s in 2 1 3; do
  $E --map_mode exact --map_sigma $s --reg $RG --reg_trust hard --reg_thr 0.62 --suffix _R62_s$s
done
$E --map_mode wrong --reg $RG --reg_trust hard --reg_thr 0.62 --suffix _R62_wrong
$E --map_mode wrong --reg $RG --reg_trust hard --reg_thr 0.75 --suffix _R75_wrong
$E --map_mode exact --map_sigma 2 --reg $RG --reg_trust hard --reg_thr 0.75 --suffix _R75_s2
