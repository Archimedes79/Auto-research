#!/bin/bash
# E318: E317 + a position scale per keypoint (--unc), 3000 steps, on the E307 maps, through the launcher with the card of experiments/E318_end-position-uncertainty (revision 2026-11-14).
# usage: start_pairs.sh <seed> [c]    DEFAULT = arm u. The control of the card is E317 arm g of the same seed (run e317_g_s<S>: same recipe without --unc, same sample order); the c arm below is the FALLBACK
#        ONLY (E317.g missing for a seed) and is NOT scheduled: the holds on E318.c_s0 / E318.c_s1 stay (one c on seed 0 would be the repo's first pair-stage noise twin: mentioned in the card, not scheduled).
# One GPU job at a time by day (D98): called by temp/e318/chain.sh; nice -n 10 as in temp/e341/start_pairs.sh.
cd "$(dirname "$0")/../.."
S=$1
MAPS=$(pwd)/runs/e307_n_s$S/model.pt
[ -f $MAPS ] || { echo "no maps $MAPS"; exit 2; }
export BFLC_SIG=1.0
if [ "$2" = "c" ]; then
  python3 ../autoresearch/tools/run.py --id E318.c_s$S --seed $S --gpu --retries 0 --require-card --split val \
    --hyp "FALLBACK control of E318 (only when E317.g of the seed is missing): the E317 recipe (geometry loss 0.5, no position scale) trained in the window of E318." \
    --pred "as E317 (snapped p >= 0.1 topo F1 +0.005 over E312)" \
    --rule "E318 card section 0: by hand (fallback control arm c); VAL 873, no TEST" \
    -- nice -n 10 bash tools/env/_pyr.sh jends_train.py --name e318_c_s$S --seed $S --maps $MAPS --e2e --scales --maps-w 1.0 --steps 3000 --geom-w 0.5 --workers 3
else
  python3 ../autoresearch/tools/run.py --id E318.u_s$S --seed $S --gpu --retries 0 --require-card --split val \
    --hyp "A learned position scale per keypoint (Laplace NLL on the end position instead of the Huber end term, detached feature of the pair logit) lets the head declare an end uncertain; the scale is informative about the true end error and marks the unsure pairs." \
    --pred "snapped p >= 0.1 topo F1 +0.005 over the control E317.g (a TIE); Spearman(sigma, end error) >= 0.3 and precision of the delivered pairs +0.03 between the confident and the unsure fifth AND more than the same split by range, as defined by temp/e318/sigma_check.py (nulls on E312 seeds 0 / 1: Spearman of range +0.12 / +0.10, fifths of range +0.17 / +0.13)" \
    --rule "E318 card section 0: by hand (jends_eval.py score --snap --thr 0.1 and score --thr 0.1, then temp/e318/sigma_check.py on the dump: Spearman and fifths as defined in the card, the fifths clause also needs the sigma split to exceed the range split of the seed); KEEP when snapped topo F1 over the control E317.g rises by more than 0.01 on both seeds with the guards held; TIE kept when the uncertainty is informative (Spearman >= 0.3 and fifths >= 0.03 on both seeds), else CUT; the kill is a non-finite loss or log sigma at the clamp for more than 90 % of the keypoints on the VAL dump; the arm is Huber -> Laplace NLL + two pair inputs + the scale, not a single factor; VAL 873, no TEST" \
    -- nice -n 10 bash tools/env/_pyr.sh jends_train.py --name e318_u_s$S --seed $S --maps $MAPS --e2e --scales --maps-w 1.0 --steps 3000 --geom-w 0.5 --unc --workers 3
fi
