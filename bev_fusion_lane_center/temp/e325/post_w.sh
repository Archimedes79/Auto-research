#!/bin/bash
# E325 post processing of one seed (CPU, niced): the snapped read-out of E312's pairs with the pieces chosen by the head (head seeds 0, 1, 2; seed 0 = the arm), then the paired scene bootstraps against the card of record. usage: post_w.sh <seed>
cd "$(dirname "$0")/../.."
S=$1; L=temp/day16/day16.log
for H in 0 1 2; do
  if [ $H = 0 ]; then WR=e325_w_s$S; TAG=e325w; else WR=e325_w_s$S:model_h$H.pt; TAG=e325w$H; fi
  nice bash tools/env/_py14.sh jends_eval.py score --run e312_p_s$S --seed $S --thr 0.1 --snap --wire head --wire-run $WR --tag $TAG >> temp/e325/score_w${H}_s$S.log 2>&1
  echo "e325 scored head $H seed $S exit $? $(date)" >> $L
done
nice bash tools/env/_py14.sh jdiag.py boot e312_p_s$S ends_card_e325w_snap_thr0.1 e312_p_s$S ends_card_snap_thr0.1 >> temp/e325/boot_w_vs_c_s$S.log 2>&1
echo "e325 boot seed $S $(date)" >> $L
