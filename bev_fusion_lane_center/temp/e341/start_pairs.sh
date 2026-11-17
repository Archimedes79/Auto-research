#!/bin/bash
# E341: the E312 pair stage (evidence ON, four inputs because the maps have no 0.5 m branch) on the maps head of E332 arm a. usage: start_pairs.sh <seed>
cd "$(dirname "$0")/../.."
. temp/e330/recipe.sh
S=$1
MAPS=$(pwd)/runs/e332_ma_s$S/model.pt
[ -f $MAPS ] || { echo "no maps $MAPS"; exit 2; }
python3 ../autoresearch/tools/run.py --id E341.p_s$S --seed $S --gpu --retries 0 --require-card --split val \
  --hyp "The 0.5 m paint branch is not needed: the evidence along the curve with its four non-paint inputs (lane heat, zone, core) on the maps head without that branch loses at most 0.01 topo F1." \
  --pred "topo F1 -0.008 against E312 of the same seed on the unsnapped read-out, -0.003 snapped; the maps head needs 1.78 instead of 3.26 GMACs at the same width" \
  --rule "E341 card section 0: evaluation/jrule_removal.py --arm e341_p_s: ADOPT when topo F1 loses at most 0.01 against E312 of the same seed on both seeds and both read-outs (unsnapped = differentiable, snapped = rule step) with the guards held (F1@1.0 <= -0.01, dup_frac <= +0.01, vertices per 100 m <= +1.0, junction F1 <= -0.03); also reported against E332 arm a; VAL 873, no TEST" \
  -- nice -n 10 bash tools/env/_pyr.sh jends_train.py --name e341_p_s$S --seed $S --maps $MAPS $RECIPE_FLAGS
