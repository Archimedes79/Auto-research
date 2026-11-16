#!/bin/bash
# E316: the E312 pair head recipe, 6000 steps, on the E307 maps, through the launcher with the card of experiments/E316_longer-pair-training.  usage: start_pairs.sh <seed>
cd "$(dirname "$0")/../.."
S=$1
MAPS=$(pwd)/runs/e307_n_s$S/model.pt
export BFLC_SIG=1.0
python3 ../autoresearch/tools/run.py --id E316.p_s$S --seed $S --gpu --retries 0 --require-card --split val \
  --hyp "The E312 pair head is still improving at step 3000 (loss falling): twice the steps on the same E307 maps lift the snapped read-out." \
  --pred "topo F1 +0.01 over E312 of the same seed on the snapped read-out at p >= 0.1" \
  --rule "E316 card section 0: by hand (jends_eval.py score --snap --thr 0.1, differences to the plain graph, then to E312); KEEP when topo F1 over E312 rises by more than 0.01 on both seeds with the guards held; TIE kept (the shorter recipe is cheaper); CUT below -0.01; VAL 873, no TEST" \
  -- bash tools/env/_pyr.sh jends_train.py --name e316_p_s$S --seed $S --maps $MAPS --e2e --scales --maps-w 1.0 --steps 6000 --workers 3
