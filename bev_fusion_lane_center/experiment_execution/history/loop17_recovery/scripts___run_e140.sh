#!/bin/bash
# E140 (E207 in the report): learned gated memory trained BEHIND the map-conditioned builder of E139 (arm B, frozen), two folds, then the held-out series fused with the model that did not see them, evaluated with exact / 2 m / no / wrong map
S=/mnt/c/ML-Experiment/scripts
L=/mnt/c/ML-Experiment/out/loop2/e17/logs
until [ -f $L/e137.done ]; do sleep 20; done
for k in 0 1; do
  bash $S/_py14.sh p17_gate.py train --fold $k --folds 2 --name e17_gateB_k$k --steps 1500 --mapnet e17_map_b > $L/e140_train_k$k.log 2>&1 &
done
wait
bash $S/_py14.sh p17_gate.py apply --names e17_gateB_k0,e17_gateB_k1 --out gateB > $L/e140_apply.log 2>&1 || exit 1
for mode in "exact 0 _m0" "exact 2 _m2" "none 0 _mnone" "wrong 0 _mwrong"; do
  set -- $mode
  bash $S/_py14.sh p17_eval.py --out e140_gateB --mapnet 1 --ckpt e17_map_b --map_mode $1 --map_sigma $2 --suffix $3 --variants gateB > $L/e140_eval$3.log 2>&1
done
echo finished > $L/e140.done
