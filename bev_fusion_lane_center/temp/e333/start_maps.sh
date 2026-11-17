#!/bin/bash
# E333 (S4) stage 1: a slimmer junction maps head, width 40 instead of 64 (about -40 % of its MACs). usage: start_maps.sh <seed> [arm, unused]
cd "$(dirname "$0")/../.."
. temp/e330/recipe.sh
S=$1
python3 ../autoresearch/tools/run.py --id E333.m_s$S --seed $S --gpu --retries 0 --require-card --split val \
  --hyp "The junction maps head is wider than it needs to be: width 40 instead of 64 loses nothing." \
  --pred "maps keypoint F1 within -0.01 of E307 of the same seed" \
  --rule "E333 card section 0: kill when the maps keypoint F1 is more than 0.05 below E307 (stage 1 only); the arm is judged after the pair stage by evaluation/jrule_removal.py" \
  -- bash tools/env/_pyr.sh jmaps_train.py --name e333_m_s$S --seed $S $MAPS_FLAGS --d 40
