#!/bin/bash
# E314: the E312 pair head recipe on the sigma 0.7 maps of E313, through the launcher with the card of experiments/E314_pairs-on-sigma-0.7-maps.  usage: start_pairs.sh <seed>
cd "$(dirname "$0")/../.."
S=$1
MAPS=$(pwd)/runs/e313_s_s$S/model.pt
export BFLC_SIG=0.7
python3 ../autoresearch/tools/run.py --id E314.p_s$S --seed $S --gpu --retries 0 --require-card --split val \
  --hyp "The E312 pair head on the heat-sigma 0.7 m maps of E313 delivers a better snapped read-out than on the sigma 1.0 m maps, because more crowded ends are found." \
  --pred "topo F1 +0.01 over E312 of the same seed on the snapped read-out at p >= 0.1" \
  --rule "E314 card section 0: by hand (jends_eval.py score --snap --thr 0.1, differences to the plain graph, then to E312); KEEP when topo F1 over E312 rises by more than 0.01 on both seeds with the guards held; TIE kept (D82); CUT below -0.01; VAL 873, no TEST" \
  -- bash tools/env/_pyr.sh jends_train.py --name e314_p_s$S --seed $S --maps $MAPS --e2e --scales --maps-w 1.0 --steps 3000 --workers 3
