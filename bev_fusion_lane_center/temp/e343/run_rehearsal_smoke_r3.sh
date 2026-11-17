#!/bin/bash
# E343 code step round 3: the rehearsal (b) of the module docstring and the --expect-code-digest line at --max 5 (smoke folder temp/e343/val_max5)
cd /mnt/c/Auto-research/bev_fusion_lane_center
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
P="nice -n 10 bash tools/env/_py14.sh jtest_eval.py"
M="--max 5"
D=temp/e343/val_max5
for S in 0 1; do $P dump-b --split val --seed $S $M; $P dump-ends --split val --seed $S --run e312_p_s$S $M; done
echo "=== dump-b to a place outside the allow-list"
$P dump-b --split val --seed 0 $M --out /tmp/not_allowed.pkl; echo "exit $?"
echo "=== E312 all"
$P all --split val --run-prefix e312_p_s --seeds 0 1 --boot 50 $M --b-dump "$D/predfull_e273_p_s{seed}.pkl"
echo "=== (b) E324 s all"
$P all --split val --test-id E343.v --run-prefix e324_s_s --seeds 0 1 --device cpu --boot 50 $M --b-dump "$D/predfull_e273_p_s{seed}.pkl" --control-ends-dump "$D/ends_e312_p_s{seed}.pkl" --control-card "$D/cards_e312/card_s{seed}.json"
DIG=$(grep -o '"code_digest": "[0-9a-f]*"' $D/cards_e324s/card_s0.json | head -1 | grep -o '[0-9a-f]\{64\}')
echo "=== (b) again with --expect-code-digest $DIG (all dumps reused under their provenance)"
$P all --split val --test-id E343.v --run-prefix e324_s_s --seeds 0 1 --device cpu --boot 50 $M --b-dump "$D/predfull_e273_p_s{seed}.pkl" --control-ends-dump "$D/ends_e312_p_s{seed}.pkl" --control-card "$D/cards_e312/card_s{seed}.json" --expect-code-digest "$DIG" --expect-code-card "$D/cards_e324s/card_s0.json"
echo "exit $?"
echo "=== (b) with a wrong digest"
$P all --split val --test-id E343.v --run-prefix e324_s_s --seeds 0 1 --device cpu --boot 50 $M --b-dump "$D/predfull_e273_p_s{seed}.pkl" --control-ends-dump "$D/ends_e312_p_s{seed}.pkl" --control-card "$D/cards_e312/card_s{seed}.json" --expect-code-digest 0123456789abcdef0123456789abcdef --expect-code-card "$D/cards_e324s/card_s0.json"
echo "exit $?"
echo "=== done"
