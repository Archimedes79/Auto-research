#!/bin/bash
# E343 code step round 4: the rehearsal (b) of the module docstring at --max 3 (smoke folder temp/e343/val_max3): the early control pass, the merged counts, the environment and the undigested-module record on real data
cd /mnt/c/Auto-research/bev_fusion_lane_center
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
P="nice -n 10 bash tools/env/_py14.sh jtest_eval.py"
M="--max 3"
D=temp/e343/val_max3
for S in 0 1; do $P dump-b --split val --seed $S $M; $P dump-ends --split val --seed $S --run e312_p_s$S $M; done
echo "=== E312 all"
$P all --split val --run-prefix e312_p_s --seeds 0 1 --boot 50 $M --b-dump "$D/predfull_e273_p_s{seed}.pkl"
echo "=== (b) E324 s all"
$P all --split val --test-id E343.v --run-prefix e324_s_s --seeds 0 1 --device cpu --boot 50 $M --b-dump "$D/predfull_e273_p_s{seed}.pkl" --control-ends-dump "$D/ends_e312_p_s{seed}.pkl" --control-card "$D/cards_e312/card_s{seed}.json"
echo "exit $?"
echo "=== (b) again: every dump reused under its provenance"
$P all --split val --test-id E343.v --run-prefix e324_s_s --seeds 0 1 --device cpu --boot 50 $M --b-dump "$D/predfull_e273_p_s{seed}.pkl" --control-ends-dump "$D/ends_e312_p_s{seed}.pkl" --control-card "$D/cards_e312/card_s{seed}.json"
echo "exit $?"
echo "=== (b) a smoke card cannot pin the code (the rehearsal card must be a full VAL card)"
$P all --split val --test-id E343.v --run-prefix e324_s_s --seeds 0 1 --device cpu --boot 50 $M --b-dump "$D/predfull_e273_p_s{seed}.pkl" --control-ends-dump "$D/ends_e312_p_s{seed}.pkl" --control-card "$D/cards_e312/card_s{seed}.json" --expect-code-card "$D/cards_e324s/card_s0.json"
echo "exit $?"
echo "=== done"
