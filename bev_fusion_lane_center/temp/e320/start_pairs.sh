#!/bin/bash
# E320: soft attachment of the connector ends to B's nodes (--attach, with the geometry loss of E317 on the final curve), 3000 steps, on the E307 maps, through the launcher with the card of experiments/E320_soft-attach-replaces-snap.  usage: start_pairs.sh <seed>
cd "$(dirname "$0")/../.."
S=$1
MAPS=$(pwd)/runs/e307_n_s$S/model.pt
export BFLC_SIG=1.0
python3 ../autoresearch/tools/run.py --id E320.a_s$S --seed $S --gpu --retries 0 --require-card --split val \
  --hyp "The snap of the read-out (a rule on the output) carries two thirds of the topology gain of E312; a learned soft attachment of the connector ends to the nodes of B's head, trained by the geometry loss on the final curve and a cross entropy over the candidates, replaces it inside the network (differentiable)." \
  --pred "topo F1 of the attached read-out at p >= 0.1 is 0.005 below E312 snapped and about 0.034 above E312 unsnapped" \
  --rule "E320 card section 0: by hand (jends_eval.py score --attach --thr 0.1, differences to the plain graph); KEEP when on both seeds topo F1 of arm a >= E312 snapped - 0.01 and > E312 unsnapped + 0.02 with the guards held; TIE kept below the first bound; CUT when not above unsnapped + 0.02; VAL 873, no TEST" \
  -- bash tools/env/_pyr.sh jends_train.py --name e320_a_s$S --seed $S --maps $MAPS --e2e --scales --maps-w 1.0 --steps 3000 --geom-w 0.5 --attach --workers 3
