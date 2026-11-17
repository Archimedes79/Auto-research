#!/bin/bash
# E334 (S5): ONE stage: the pair head and the junction maps head trained jointly from the untrained head (no separate maps stage). usage: start_pairs.sh <seed> [o3|o6]   (o3: 3000 steps, run E334.o_s<S>; o6: 6000 steps, E334.o6_s<S>, the conditional arm of the card)
cd "$(dirname "$0")/../.."
. temp/e330/recipe.sh
S=$1; V=${2:-o3}
INIT=$(pwd)/runs/e334_init_s$S/model.pt
[ -f $INIT ] || bash tools/env/_py14.sh e334/make_init.py $S
if [ "$V" = "o6" ]; then ID=E334.o6_s$S; NAME=e334_o6_s$S; STEPS="--steps 6000"; else ID=E334.o_s$S; NAME=e334_o_s$S; STEPS=""; fi
python3 ../autoresearch/tools/run.py --id $ID --seed $S --gpu --retries 0 --require-card --split val \
  --hyp "The two-stage training (maps 3000 steps, then maps + pairs jointly 3000 steps) is not needed: one joint stage from the untrained head gives the same delivered graph." \
  --pred "topo F1 within -0.01 of E312 of the same seed at 3000 steps (half the maps training); equal with 6000 steps" \
  --rule "E334 card section 0: evaluation/jrule_removal.py --arm e334_o_s (o6: e334_o6_s): ADOPT when topo F1 loses at most 0.01 against E312 of the same seed on both seeds and both read-outs with the guards held; VAL 873, no TEST" \
  -- bash tools/env/_pyr.sh jends_train.py --name $NAME --seed $S --maps $INIT $RECIPE_FLAGS $STEPS
