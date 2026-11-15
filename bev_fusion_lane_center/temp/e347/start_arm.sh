#!/bin/bash
# E347 stage 1, one arm and seed through the launcher with the card (GPU slot): usage: start_arm.sh <g|o> <seed>   (D101: --workers 4; BFLC_SIG=1.0 is exported by run_staged_gpu.sh)
cd "$(dirname "$0")/../.."
A=$1; S=$2
case $A in
  g) FL="--r2"; HYP="E347 arm g: the junction maps head with its output at 2 m (50 x 50 grid, sub-cell offset, Gaussian heat sigma 1.0 m, 3 x 3 peak read-out) carries the ends as well as the 1 m head without the paint branch (E332 arm a)";;
  o) FL="--r2 --onehot"; HYP="E347 arm o: the 2 m maps head with a ONE-TO-ONE end target (focal loss, threshold read-out, no peak step) carries the ends as well as the 1 m head without the paint branch, without duplicates";;
esac
python3 ../autoresearch/tools/run.py --id E347.${A}_s$S --seed $S --gpu --ram-mb 6000 --retries 0 --require-card --split val \
  --hyp "$HYP" \
  --pred "arm g: keypoint F1 at 3 m -0.02, crowded entry recall -0.10 against E332 arm a (INCONCLUSIVE or worse in the peak read-out); arm o: keypoint F1 at 3 m -0.015, at 1.5 m -0.03, crowded -0.03, duplicates within the bound" \
  --rule "E347 card section 0: stage 1 is a label and a kill gate (by hand: maps scorer): PASS when keypoint F1 at 3 m is at most 0.01 below c, at 1.5 m at most 0.02 below, crowded entry recall at most 0.04 below, arm o also the duplicate clause, on both seeds at each run's own best threshold; FAIL: more than 0.03 / 0.05 below on both seeds" \
  -- env BFLC_SIG=1.0 bash temp/e347/run_staged_gpu.sh jmaps_train.py --name e347_${A}_s$S --seed $S --ctx dil --sigma 1.0 --steps 3000 --workers 4 --no-paint $FL
