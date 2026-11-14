#!/bin/bash
# E317: the E312 pair head recipe + the geometry loss of the delivered connector (--geom-w 0.5, no attachment), 3000 steps, on the E307 maps, through the launcher with the card of experiments/E317_connector-geometry-loss (revision 2026-11-14).
# usage: start_pairs.sh <seed>.  One GPU job at a time by day (D98): called by temp/e317/chain.sh; nice -n 10 as in temp/e341/start_pairs.sh.
cd "$(dirname "$0")/../.."
S=$1
MAPS=$(pwd)/runs/e307_n_s$S/model.pt
[ -f $MAPS ] || { echo "no maps $MAPS"; exit 2; }
export BFLC_SIG=1.0
python3 ../autoresearch/tools/run.py --id E317.g_s$S --seed $S --gpu --retries 0 --require-card --split val \
  --hyp "The delivered connector has no loss of its own (only the keypoints, the pair logit and the scales are trained): a Chamfer / smooth-L1 loss of the Bezier curve against the GT connector polyline and of the keypoint ends against the GT ends makes the shape a constraint; E320 trained it together with the attachment (own keypoints +0.006 / +0.008 unsnapped over E312), this run isolates the loss without the attachment." \
  --pred "snapped p >= 0.1 topo F1 +0.005 over E312 of the same seed (a TIE); unsnapped p >= 0.1 topo F1 +0.006 over E312; snapped with the keypoint geometry (--snap --snap-nogeom) +0.008 over u' (E312 0.6706 / 0.6760)" \
  --rule "E317 card section 0: by hand (jends_eval.py score --snap --thr 0.1 = the rule's read-out, score --thr 0.1 unsnapped, score --snap --snap-nogeom --thr 0.1 for the end-geometry claim; differences to the plain graph, then to E312 and, as context only, to E324 s); KEEP when snapped topo F1 over E312 rises by more than 0.01 on both seeds with the guards held; TIE kept; CUT below -0.01; the end-geometry claim holds when snap-nogeom topo F1 is at least u' + 0.005 on both seeds; the kill is a non-finite loss, maps keypoint F1 against E307 is reported (flag below -0.08); VAL 873, no TEST" \
  -- nice -n 10 bash tools/env/_pyr.sh jends_train.py --name e317_g_s$S --seed $S --maps $MAPS --e2e --scales --maps-w 1.0 --steps 3000 --geom-w 0.5 --workers 3
