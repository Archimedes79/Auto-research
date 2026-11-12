#!/bin/bash
# E298 on the E312 system: the pair head (--e2e --scales) on the E307 maps with the driven-pair weight, through the launcher with the card of experiments/E298_driven-pair-weight (amendment of 2026-11-12).  usage: start_driven_e312.sh <seed>
cd "$(dirname "$0")/../.."
S=$1
MAPS=$(pwd)/runs/e307_n_s$S/model.pt
export BFLC_SIG=1.0
python3 ../autoresearch/tools/run.py --id E298.w_s$S --seed $S --gpu --retries 0 --require-card --split val \
  --hyp "Weighting the positive pair terms of the pairs the ego drives (GT label tier 1, w = 2) raises the recall of the driven pairs of the E312 pair head without hurting the overall pair recall, precision or the snapped read-out." \
  --pred "driven pair recall +0.06 over E312 of the same seed; overall pair recall / precision within 0.02" \
  --rule "E298 card section 0 + amendment: by hand (jsplits.py driven split + jends_eval.py score --snap --thr 0.1); KEEP when the driven pair recall rises by more than 0.03 on both seeds with pair recall / precision within 0.02 and the snapped topo F1 within 0.01; TIE kept and refined (D82); CUT when a guard breaks on both seeds; VAL 873, no TEST" \
  -- bash tools/env/_pyr.sh jends_train.py --name e298_w_s$S --seed $S --maps $MAPS --e2e --scales --driven-w 2 --maps-w 1.0 --steps 3000 --workers 3
