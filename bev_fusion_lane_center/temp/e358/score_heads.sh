#!/bin/bash
# E358: the end-to-end read-out scored with the heads of one seed on the STAGED scorer (CPU, nice, 2 threads), then the paired scene bootstrap against the stored PRIMARY card.  usage: score_heads.sh <seed>
# heads 0, 1, 2 (tags e358w, e358w1, e358w2) and the no-jitter ablation (tag e358wn); a failed scoring stops the script (a tagged card is never overwritten: a rerun needs the cards moved away by hand or new tags)
cd "$(dirname "$0")/../.."
S=$1
[ -n "$S" ] || { echo "usage: score_heads.sh <seed>"; exit 2; }
sha256sum temp/_session_scratch_e358/stage/*.py
for H in 0 1 2 n; do
  case $H in 0) WR=e358_w_s$S; TAG=e358w;; n) WR=e358_w_s$S:model_wn.pt; TAG=e358wn;; *) WR=e358_w_s$S:model_h$H.pt; TAG=e358w$H;; esac
  bash temp/e358/run_staged.sh jends_eval.py score --run e324_s_s$S --seed $S --thr 0.1 --attach --snap --snap-nogeom --wire head --wire-run $WR --tag $TAG || exit 1
done
nice -n 19 bash tools/env/_py14.sh jdiag.py boot e324_s_s$S ends_card_e358w_snap_nogeom_attach_thr0.1 e324_s_s$S ends_card_snap_nogeom_attach_thr0.1 > temp/e358/boot_w_vs_c_s$S.log 2>&1
echo "score_heads seed $S done $(date)"
