#!/bin/bash
# E284: the junction end head (phase 8) on the frozen B run of one seed, through the launcher with the card of experiments/E284_junction-end-head.  usage: start_ends.sh <seed> [smoke]
cd "$(dirname "$0")/../.."
S=$1
if [ "$2" = "smoke" ]; then
  python3 ../autoresearch/tools/run.py --id E284.smoke2_s$S --seed $S --gpu --retries 0 --no-card --split val \
    -- bash tools/env/_pyr.sh qgraph_train.py --name e284_smoke2_s$S --seed $S --phases 8 --resume-from ../../runs/e273_p_s$S/model.pt --workers 3 --smoke
  exit $?
fi
python3 ../autoresearch/tools/run.py --id E284.h_s$S --seed $S --gpu --retries 0 --require-card --split val \
  --hyp "A tolerant junction end head (entry / exit keypoints with a Gaussian target of 2 m, offset, heading, and a pair decision), trained on the frozen stages of the lane head B, delivers connectors that raise the topo F1 of the delivered graph (B + hard step 1.5 m) when they are added and the delivered long links they cover are replaced." \
  --pred "topo F1 +0.010 over the plain delivered graph of the same seed (a TIE), lanes found +0.07, entry / exit recall about 0.70, pair recall about 0.35 (E283 section 8: K3 +0.013 / +0.016, K4 +0.006 / +0.009)" \
  --rule "E284 card section 0 (D79): KEEP when topo F1 > +0.02 on both seeds with F1@1.0 >= -0.01, dup_frac <= +0.02, vertices per 100 m <= +1.0, junction F1 >= -0.03 against the plain delivered graph of the same seed; CUT below -0.02 or a guard broken on both seeds; else TIE; by hand (qends_eval.py score), VAL 873, final checkpoint, threshold 0.5 fixed beforehand, no TEST" \
  -- bash tools/env/_pyr.sh qgraph_train.py --name e284_h_s$S --seed $S --phases 8 --resume-from ../../runs/e273_p_s$S/model.pt --steps 4000 --workers 3
