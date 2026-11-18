#!/bin/bash
# E134: learned gated memory in front of the frozen builder: 2 folds trained in parallel, then the held-out series fused with the model that did not see them, then evaluated
S=/mnt/c/ML-Experiment/scripts
L=/mnt/c/ML-Experiment/out/loop2/e17/logs
for k in 0 1; do
  bash $S/_py14.sh p17_gate.py train --fold $k --folds 2 --name e17_gate_k$k --steps 1500 > $L/e134_train_k$k.log 2>&1 &
done
wait
bash $S/_py14.sh p17_gate.py apply --names e17_gate_k0,e17_gate_k1 --out gate > $L/e134_apply.log 2>&1 || exit 1
bash $S/_py14.sh p17_eval.py --out e134_gate --prev 0 --variants gate > $L/e134.log 2>&1
echo finished > $L/e134.done
