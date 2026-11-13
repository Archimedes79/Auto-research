#!/bin/bash
# E312 pair head on the E307 maps (sigma 1.0) (maps and pairs trained together, the recipe of E305, + the learned handle scales of E303), BFLC_SIG=1.0 so that the heat targets of the joint training match the maps.  usage: start_pairs.sh <seed>
cd "$(dirname "$0")/../.."
S=$1
MAPS=$(pwd)/runs/e307_n_s$S/model.pt
export BFLC_SIG=1.0
python3 ../autoresearch/tools/run.py --id E312.p_s$S --seed $S --gpu --retries 0 --require-card --split val \
  --hyp "The E305 pair head (maps and pair head trained together) with the E303 handle scales on the narrower, far-weighted maps m2 delivers a better snapped read-out than the E305 head on the maps of E286 j1." \
  --pred "topo F1 +0.01 over E305 of the same seed on the snapped read-out at p >= 0.1" \
  --rule "E312 card section 0: by hand (jends_eval.py score --snap --thr 0.1, differences to the plain graph, then to E305); KEEP when topo F1 over E305 rises by more than 0.01 on both seeds with the guards held; TIE kept and refined (D82); CUT below; VAL 873, no TEST" \
  -- bash tools/env/_pyr.sh jends_train.py --name e312_p_s$S --seed $S --maps $MAPS --e2e --scales --maps-w 1.0 --steps 3000 --workers 3
