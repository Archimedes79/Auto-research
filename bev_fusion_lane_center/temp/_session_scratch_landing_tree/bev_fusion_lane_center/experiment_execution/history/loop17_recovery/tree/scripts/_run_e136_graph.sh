#!/bin/bash
# E136: graph quality of the joint arms (M = 192, K0 = 96, recursion off) on the dark / blur series, single frame and hold
S=/mnt/c/ML-Experiment/scripts
L=/mnt/c/ML-Experiment/out/loop2/e17/logs
V=single,maxdecay:0.95
bash $S/_py14.sh p17_eval.py --out e136_graph_base --set M=192 K0=96 --prev 0 --perts dark,blur --variants $V > $L/e136_graph_base.log 2>&1
for arm in a b c; do
  bash $S/_py14.sh p17_eval.py --out e136_graph_$arm --set M=192 K0=96 --prev 0 --ckpt e17_joint_$arm --perts dark,blur --variants $V > $L/e136_graph_$arm.log 2>&1
done
echo finished > $L/e136_graph.done
