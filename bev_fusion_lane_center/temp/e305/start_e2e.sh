#!/bin/bash
# E305: the junction maps head trained together with the pair head, through the launcher with the card of experiments/E305_e2e-maps-and-pairs.  usage: start_e2e.sh <e|e3> <seed> <maps arm j0|j1|j2> [smoke]
cd "$(dirname "$0")/../.."
ARM=$1; S=$2; MA=${3:-j1}
W=1.0; [ "$ARM" = "e3" ] && W=0.3
MAPS=runs/e286_${MA}_s$S/model.pt
if [ "$4" = "smoke" ]; then
  python3 ../autoresearch/tools/run.py --id E305.smoke_${ARM}_s$S --seed $S --gpu --retries 0 --no-card --split val \
    -- bash tools/env/_pyr.sh jends_train.py --name e305_smoke_${ARM}_s$S --seed $S --maps $MAPS --e2e --maps-w $W --smoke --workers 3
  exit $?
fi
python3 ../autoresearch/tools/run.py --id E305.${ARM}_s$S --seed $S --gpu --retries 0 --require-card --split val \
  --hyp "Training the junction maps head together with the pair head (the pair loss reaches the heat through the soft-argmax positions and the features) beats the pair head on frozen maps, or at least ties it while the chain is end to end." \
  --pred "topo F1 +0.01 over E287 p2 of the same seed (a TIE by the band); maps keypoint F1 within 0.02 of E286 ${MA}" \
  --rule "E305 card section 0: by hand (jends_eval.py score, D79 guards); KEEP when topo F1 rises by more than 0.01 on both seeds with the guards held; TIE kept and refined (D82); CUT below or the kill; VAL 873, no TEST" \
  -- bash tools/env/_pyr.sh jends_train.py --name e305_${ARM}_s$S --seed $S --maps $MAPS --e2e --maps-w $W --steps 3000 --workers 3
