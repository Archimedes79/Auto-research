#!/bin/bash
# E343 code step round 2: the rehearsal command lines of the module docstring at --max 6 (smoke folder temp/e343/val_max6): (a) control from the stored VAL files, (b) control from the same fresh dumps
cd /mnt/c/Auto-research/bev_fusion_lane_center
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
P="nice -n 10 bash tools/env/_py14.sh jtest_eval.py"
M="--max 6"
for S in 0 1; do $P dump-b --split val --seed $S $M; $P dump-ends --split val --seed $S --run e312_p_s$S $M; done
echo "=== E312 all (cards of the fresh dumps)"
$P all --split val --run-prefix e312_p_s --seeds 0 1 --boot 50 $M --b-dump 'temp/e343/val_max6/predfull_e273_p_s{seed}.pkl'
echo "=== (a) control from the stored VAL files"
$P all --split val --test-id E343.v --run-prefix e324_s_s --seeds 0 1 --device cpu --boot 50 $M --b-dump 'temp/e343/val_max6/predfull_e273_p_s{seed}.pkl' --control-ends-dump 'runs/e312_p_s{seed}/ends_val.pkl' --control-card 'temp/e343/val/val_check/verify_card_e312_p_s{seed}.json'
echo "=== (b) control from the same fresh dumps"
$P all --split val --test-id E343.v --run-prefix e324_s_s --seeds 0 1 --device cpu --boot 50 $M --b-dump 'temp/e343/val_max6/predfull_e273_p_s{seed}.pkl' --control-ends-dump 'temp/e343/val_max6/ends_e312_p_s{seed}.pkl' --control-card 'temp/e343/val_max6/cards_e312/card_s{seed}.json'
echo "=== done"
