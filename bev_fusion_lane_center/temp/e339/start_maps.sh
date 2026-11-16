#!/bin/bash
# E339 stage 1: the junction maps head of width 40 (the width of E333) taught by the E307 head of the same seed (soft-target distillation). usage: start_maps.sh <seed> [arm, unused]
cd "$(dirname "$0")/../.."
. temp/e330/recipe.sh
S=$1
python3 ../autoresearch/tools/run.py --id E339.m_s$S --seed $S --gpu --retries 0 --require-card --split val \
  --hyp "A narrower junction maps head taught by the wide one (soft-target distillation) loses nothing." \
  --pred "maps keypoint F1 within -0.02 of E307 of the same seed" \
  --rule "E339 card section 0: kill when the maps keypoint F1 is more than 0.05 below E307 (stage 1 only); the arm is judged after the pair stage by evaluation/jrule_removal.py" \
  -- bash tools/env/_pyr.sh jmaps_train.py --name e339_m_s$S --seed $S $MAPS_FLAGS --d 40 --teacher $(pwd)/runs/e307_n_s$S/model.pt --distill-w 1.0
