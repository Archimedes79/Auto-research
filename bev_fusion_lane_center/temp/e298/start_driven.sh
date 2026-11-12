#!/bin/bash
# E298: the pair head p2 with the driven-pair weight (w = 2) on the frozen junction maps of E286, through the launcher with the card of experiments/E298_driven-pair-weight.
# usage: start_driven.sh <seed> <maps arm j0|j1|j2> [smoke]
cd "$(dirname "$0")/../.."
S=$1; MAPS=$2
R=$(pwd)
MP=$R/runs/e286_${MAPS}_s$S/model.pt
if [ "$3" = "smoke" ]; then
  python3 ../autoresearch/tools/run.py --id E298.smoke_s$S --seed $S --gpu --retries 0 --no-card --split val \
    -- bash tools/env/_pyr.sh jends_train.py --name e298_smoke_s$S --seed $S --maps $MP --driven-w 2 --smoke --workers 3
  exit $?
fi
python3 ../autoresearch/tools/run.py --id E298.w_s$S --seed $S --gpu --retries 0 --require-card --split val \
  --hyp "Weighting the positive pair terms of the pairs the ego drives (GT label tier 1, w = 2) raises the recall of the driven pairs of the pair head p2 without hurting the overall pair recall, precision or topo F1." \
  --pred "driven pair recall +0.06 over E287 p2 of the same seed; overall pair recall / precision within 0.02" \
  --rule "E298 card section 0: by hand (jsplits.py driven split + jends_eval.py score); KEEP when the driven pair recall rises by more than 0.03 on both seeds with pair recall / precision within 0.02 and topo F1 within 0.01; TIE kept and refined (D82); CUT when a guard breaks on both seeds; VAL 873, no TEST" \
  -- bash tools/env/_pyr.sh jends_train.py --name e298_w_s$S --seed $S --maps $MP --driven-w 2 --steps 3000 --workers 3
