#!/bin/bash
# E318: E317 + a position scale per keypoint (--unc), 3000 steps, on the E307 maps, through the launcher with the card of experiments/E318_end-position-uncertainty.
# usage: start_pairs.sh <seed> [c]      c = the control arm of the card when E317 was not run first (--geom-w 0.5 without --unc, run id E318.c_s<seed>)
cd "$(dirname "$0")/../.."
S=$1
MAPS=$(pwd)/runs/e307_n_s$S/model.pt
export BFLC_SIG=1.0
if [ "$2" = "c" ]; then
  python3 ../autoresearch/tools/run.py --id E318.c_s$S --seed $S --gpu --retries 0 --require-card --split val \
    --hyp "Control of E318: the E317 recipe (geometry loss 0.5, no position scale) trained in the window of E318." \
    --pred "as E317 (snapped p >= 0.1 topo F1 +0.005 over E312)" \
    --rule "E318 card section 0: by hand (control arm c); VAL 873, no TEST" \
    -- bash tools/env/_pyr.sh jends_train.py --name e318_c_s$S --seed $S --maps $MAPS --e2e --scales --maps-w 1.0 --steps 3000 --geom-w 0.5 --workers 3
else
  python3 ../autoresearch/tools/run.py --id E318.u_s$S --seed $S --gpu --retries 0 --require-card --split val \
    --hyp "A learned position scale per keypoint (Laplace NLL on the end position, detached feature of the pair logit) lets the head declare an end uncertain; the scale is informative about the true end error and marks the unsure pairs." \
    --pred "snapped p >= 0.1 topo F1 +0.005 over the control (a TIE); Spearman(sigma, end error) >= 0.3; precision of the delivered pairs +0.03 between the confident and the unsure fifth" \
    --rule "E318 card section 0: by hand (jends_eval.py score --snap --thr 0.1 and score --thr 0.1, then the Spearman / precision check on the dump's logsig); KEEP when snapped topo F1 over the control rises by more than 0.01 on both seeds with the guards held; TIE kept when the uncertainty is informative, else CUT; VAL 873, no TEST" \
    -- bash tools/env/_pyr.sh jends_train.py --name e318_u_s$S --seed $S --maps $MAPS --e2e --scales --maps-w 1.0 --steps 3000 --geom-w 0.5 --unc --workers 3
fi
