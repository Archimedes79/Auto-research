#!/bin/bash
# E332 (S3) stage 2: the E312 pair stage (no evidence: it reads the zone / core / paint maps) on the reduced maps. usage: start_pairs.sh <seed> <a|b>
cd "$(dirname "$0")/../.."
. temp/e330/recipe.sh
S=$1; A=$2
MAPS=$(pwd)/runs/e332_m${A}_s$S/model.pt
python3 ../autoresearch/tools/run.py --id E332.p${A}_s$S --seed $S --gpu --retries 0 --require-card --split val \
  --hyp "The delivered graph does not need the auxiliary maps: the E312 pair stage without evidence on maps without the paint branch (a) / without zone, core and paint (b) loses nothing." \
  --pred "topo F1 within -0.005 of E312 of the same seed on both read-outs; the maps head needs 1.78 instead of 3.26 GMACs" \
  --rule "E332 card section 0: evaluation/jrule_removal.py --arm e332_p${A}_s: ADOPT when topo F1 loses at most 0.01 against E312 of the same seed on both seeds and both read-outs (unsnapped = differentiable, snapped = rule step) with the guards held; also reported against E331 (the same without evidence on the full maps); VAL 873, no TEST" \
  -- bash tools/env/_pyr.sh jends_train.py --name e332_p${A}_s$S --seed $S --maps $MAPS --no-evidence $RECIPE_FLAGS
