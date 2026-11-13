#!/bin/bash
# E317: the E312 pair head recipe + the geometry loss of the delivered connector (--geom-w 0.5), 3000 steps, on the E307 maps, through the launcher with the card of experiments/E317_connector-geometry-loss.  usage: start_pairs.sh <seed>
cd "$(dirname "$0")/../.."
S=$1
MAPS=$(pwd)/runs/e307_n_s$S/model.pt
export BFLC_SIG=1.0
python3 ../autoresearch/tools/run.py --id E317.g_s$S --seed $S --gpu --retries 0 --require-card --split val \
  --hyp "The delivered connector has no loss of its own (only the keypoints, the pair logit and the scales are trained): a Chamfer / smooth-L1 loss of the Bezier curve against the GT connector polyline and of the keypoint ends against the GT ends makes the shape a constraint, lifts the unsnapped read-out and closes part of the gap to the snapped one." \
  --pred "snapped p >= 0.1 topo F1 +0.005 over E312 of the same seed (a TIE); unsnapped p >= 0.1 topo F1 +0.015 over E312" \
  --rule "E317 card section 0: by hand (jends_eval.py score --snap --thr 0.1 and score --thr 0.1, differences to the plain graph, then to E312); KEEP when snapped topo F1 over E312 rises by more than 0.01 on both seeds with the guards held; TIE kept; CUT below -0.01; unsnapped reading in the card; VAL 873, no TEST" \
  -- bash tools/env/_pyr.sh jends_train.py --name e317_g_s$S --seed $S --maps $MAPS --e2e --scales --maps-w 1.0 --steps 3000 --geom-w 0.5 --workers 3
