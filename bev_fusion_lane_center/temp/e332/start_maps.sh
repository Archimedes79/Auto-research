#!/bin/bash
# E332 (S3) stage 1: the junction maps head without its auxiliary maps. arm a: no 0.5 m paint branch; arm b: also no zone / core supervision. usage: start_maps.sh <seed> <a|b>
cd "$(dirname "$0")/../.."
. temp/e330/recipe.sh
S=$1; A=$2
NOZONE=""; [ "$A" = "b" ] && NOZONE="--no-zone"
python3 ../autoresearch/tools/run.py --id E332.m${A}_s$S --seed $S --gpu --retries 0 --require-card --split val \
  --hyp "The junction maps head does not need its auxiliary maps (the paint branch is 45 % of its MACs; zone / core): without the evidence along the curve nothing reads them." \
  --pred "maps keypoint F1 within -0.01 of E307 of the same seed" \
  --rule "E332 card section 0: kill when the maps keypoint F1 is more than 0.05 below E307 (stage 1 only); the arm is judged after the pair stage by evaluation/jrule_removal.py" \
  -- bash tools/env/_pyr.sh jmaps_train.py --name e332_m${A}_s$S --seed $S $MAPS_FLAGS --no-paint $NOZONE
