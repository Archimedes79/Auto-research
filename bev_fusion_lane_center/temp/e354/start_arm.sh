#!/bin/bash
# E354: one training run of the card (experiments/E354_recursive-feature-state/experiment.md) through run.py.
# usage: start_arm.sh <r|z> <seed>     r = state on, z = state off (the equal-budget control)
# K = 2, 4 sequences (8 frames per step): fixed by the launch gate of 2026-11-15 (card section 0). Never edit this script while a run uses it.
cd "$(dirname "$0")/../.."
export BFLC_SIG=1.0
ARM=$1; S=$2
case $ARM in r) STATE=on;; z) STATE=off;; *) echo "arm r or z"; exit 2;; esac
MAPS=$(pwd)/runs/e332_ma_s$S/model.pt
[ -f $MAPS ] || { echo "no maps $MAPS"; exit 2; }
[ -e runs/e354_${ARM}_s$S ] && { echo "runs/e354_${ARM}_s$S exists"; exit 2; }
python3 ../autoresearch/tools/run.py --id E354.${ARM}_s$S --seed $S --gpu --ram-mb 6000 --retries 0 --require-card --split val \
  --hyp "A learned ego-warped feature state of the previous keyframe (one gated convolution in the 2 m trunk of the junction maps head, gradient through the state, D103) costs no accuracy against the same training with the state empty and lowers the pair flip rate between keyframes." \
  --pred "unsnapped topo F1 of r minus z +0.004 on both seeds (a TIE), snapped +0.003; pair flip rate at matched delivery -3 points (from 35.2 / 33.2 % on E341); +74.4 k parameters, +0.186 GMACs" \
  --rule "E354 card section 0 (by hand: replay): KEEP = r at least 0.01 above z on both seeds unsnapped, snapped not below z by more than 0.005, guards held, not below E341 by more than 0.01; TIE = not KEEP and not below z by more than 0.01 on both seeds and both read-outs (status 'offered as the D103 solution' only with the matched-delivery flip-rate clause and the nulls on r's own checkpoint); CUT = below z by more than 0.01 on both seeds, a guard broken on both, or the kill; else INCONCLUSIVE; VAL 873, no TEST" \
  -- bash tools/env/_pyr.sh jrecur_train.py --name e354_${ARM}_s$S --seed $S --maps $MAPS --e2e --scales --maps-w 1.0 --steps 3000 --seq 2 --bs 4 --state $STATE --workers 4
