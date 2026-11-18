#!/bin/bash
# E133b: failure-frame fine-tuning of the champion, 2 folds over the VAL failure-series scenes (scene-disjoint), then held-out evaluation of each fold
S=/mnt/c/ML-Experiment/scripts
L=/mnt/c/ML-Experiment/out/loop2/e17/logs
for k in 0 1; do
  bash $S/_py14.sh p17_ft.py --name e17_ft_k$k --fold $k --folds 2 --u 1.5 --lr 3e-4 > $L/e133b_train_k$k.log 2>&1 &
done
wait
for k in 0 1; do
  bash $S/_py14.sh p17_eval.py --out e133b_k$k --ckpt e17_ft_k$k --variants single,maxdecay:0.95,ema_hold_mon:0.5 > $L/e133b_eval_k$k.log 2>&1
done
echo finished > $L/e133b.done
