#!/bin/bash
# E287: the junction pair head v2 on the frozen junction maps of E286, through the launcher with the card of experiments/E287_junction-pair-head-v2.
# usage: start_pairs.sh <p0|p1|p2> <seed> <maps arm j0|j1|j2> [smoke]
cd "$(dirname "$0")/../.."
ARM=$1; S=$2; MAPS=$3
R=$(pwd)
case $ARM in p0) FLAGS="--no-order --no-evidence --cross-w 0";; p1) FLAGS="--no-evidence";; p2|q2) FLAGS="";; *) echo "arm?"; exit 2;; esac
MP=$R/runs/e286_${MAPS}_s$S/model.pt
if [ "$4" = "smoke" ]; then
  python3 ../autoresearch/tools/run.py --id E287.smoke_${ARM}_s$S --seed $S --gpu --retries 0 --no-card --split val \
    -- bash tools/env/_pyr.sh jends_train.py --name e287_smoke_${ARM}_s$S --seed $S --maps $MP $FLAGS --smoke --workers 3
  exit $?
fi
python3 ../autoresearch/tools/run.py --id E287.${ARM}_s$S --seed $S --gpu --retries 0 --require-card --split val \
  --hyp "A pair head on the keypoints of the junction maps (E286, arm ${MAPS}) with the order rule (rank features + a soft crossing penalty) and the evidence along the Hermite curve delivers connectors that raise the topo F1 of the delivered graph (B + hard step 1.5 m) over the plain graph; arm ${ARM}: flags '${FLAGS}'." \
  --pred "topo F1 +0.015 over the plain delivered graph of the same seed (a TIE by the rule), pair recall 0.45 / precision 0.70, end recall 0.70 (E284: -0.004, 0.31 / 0.64, 0.58)" \
  --rule "E287 card section 0 (D79): KEEP when topo F1 > +0.02 on both seeds with F1@1.0 >= -0.01, dup_frac <= +0.02, vertices per 100 m <= +1.0, junction F1 >= -0.03 against the plain delivered graph of the same seed; CUT below -0.02 or a guard broken on both seeds; else TIE (kept and refined, D82); by hand (jends_eval.py score), VAL 873, final checkpoint, threshold 0.5 fixed beforehand, no TEST" \
  -- bash tools/env/_pyr.sh jends_train.py --name e287_${ARM}_s$S --seed $S --maps $MP $FLAGS --steps 3000 --workers 3
