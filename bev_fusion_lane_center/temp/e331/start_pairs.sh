#!/bin/bash
# E331 (S1): the E312 pair stage WITHOUT the evidence along the curve (judged on the unsnapped and the snapped read-out, card E331). usage: start_pairs.sh <seed>
cd "$(dirname "$0")/../.."
. temp/e330/recipe.sh
S=$1
MAPS=$(pwd)/runs/${BASE_MAPS_PREFIX}$S/model.pt
python3 ../autoresearch/tools/run.py --id E331.e_s$S --seed $S --gpu --retries 0 --require-card --split val \
  --hyp "The pair head does not need the evidence along the curve (E287: p2 over p1 added nothing): removing it simplifies the head, its inputs and the code at no loss." \
  --pred "topo F1 within -0.005 of E312 of the same seed on both read-outs (a TIE: the cheaper variant)" \
  --rule "E331 card section 0: evaluation/jrule_removal.py --arm e331_e_s: ADOPT when topo F1 loses at most 0.01 against E312 of the same seed on both seeds and on both read-outs (unsnapped = differentiable, snapped = rule step) with the guards held; VAL 873, no TEST" \
  -- bash tools/env/_pyr.sh jends_train.py --name e331_e_s$S --seed $S --maps $MAPS --no-evidence $RECIPE_FLAGS
