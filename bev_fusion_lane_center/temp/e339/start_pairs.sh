#!/bin/bash
# E339 stage 2: the E312 pair stage (evidence on, as E312) on the taught narrow maps. usage: start_pairs.sh <seed> [arm, unused]
cd "$(dirname "$0")/../.."
. temp/e330/recipe.sh
S=$1
MAPS=$(pwd)/runs/e339_m_s$S/model.pt
python3 ../autoresearch/tools/run.py --id E339.p_s$S --seed $S --gpu --retries 0 --require-card --split val \
  --hyp "A narrower head taught by the wide one gives the same delivered graph at a fraction of the MACs." \
  --pred "topo F1 within -0.006 of E312 of the same seed on both read-outs" \
  --rule "E339 card section 0: evaluation/jrule_removal.py --arm e339_p_s: ADOPT when topo F1 loses at most 0.01 against E312 of the same seed on both seeds and both read-outs with the guards held; VAL 873, no TEST" \
  -- bash tools/env/_pyr.sh jends_train.py --name e339_p_s$S --seed $S --maps $MAPS $RECIPE_FLAGS
