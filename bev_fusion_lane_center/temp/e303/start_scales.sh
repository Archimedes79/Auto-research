#!/bin/bash
# E303: the pair head p2 with two learned handle scales per pair (the delivered connector curve), on the frozen junction maps of E286, through the launcher with the card of experiments/E303_connector-handle-scales.
# usage: start_scales.sh <seed> <maps arm j0|j1|j2> [smoke]
cd "$(dirname "$0")/../.."
S=$1; MAPS=$2
R=$(pwd)
MP=$R/runs/e286_${MAPS}_s$S/model.pt
if [ "$3" = "smoke" ]; then
  python3 ../autoresearch/tools/run.py --id E303.smoke_s$S --seed $S --gpu --retries 0 --no-card --split val \
    -- bash tools/env/_pyr.sh jends_train.py --name e303_smoke_s$S --seed $S --maps $MP --scales --smoke --workers 3
  exit $?
fi
python3 ../autoresearch/tools/run.py --id E303.s_s$S --seed $S --gpu --retries 0 --require-card --split val \
  --hyp "Two learned Hermite handle scales per pair (trained against the scales fitted to the GT connector) make the delivered connector curves closer to the true connectors than the chord tangent, which is within 1 m of the GT connector for only 77 - 89 % of the points beyond 5 m depth (E293), and raise F1@1.0 of the delivered graph." \
  --pred "F1@1.0 +0.01 over the chord-tangent connectors of the same pairs (same dump scored with and without --chord); topo F1 and duplicates unchanged within the guards" \
  --rule "E303 card section 0: by hand (jends_eval.py score with and without --chord on the same dump); KEEP when F1@1.0 rises by more than 0.01 on both seeds with topo F1 within 0.01, dup_frac within 0.02, vertices within 1.0; TIE kept and refined (D82); CUT below; VAL 873, no TEST" \
  -- bash tools/env/_pyr.sh jends_train.py --name e303_s_s$S --seed $S --maps $MP --scales --steps 3000 --workers 3
