#!/bin/bash
# E317 chain (D96 / D98): the pair stage with the geometry loss (temp/e317/start_pairs.sh) for the two seeds ONE AFTER THE OTHER (one GPU job at a time, nice), per seed the post-processing of temp/e330/chain_lib.sh
# (dump, unsnapped / snapped / snap-nogeom scores, the plain card, pair PR against E312, maneuver coverage, the joint maps' own score), then the read-outs of the card: E312's unsnapped card WITH counts (tag cnt),
# the end error on VAL, the maps flag against E307 (report-only), and the paired scene bootstraps against E312 on all three read-outs and, as context only (D94), against E324 s.
# usage: chain.sh [marker file to wait for]. Starts only when chain_lib.sh can_start (nights, Mon-Fri 08:30-15:30, weekend days since D98) and host_ok (free RAM >= 8 GB, load < 8; 6 / 10 until Mon 2026-11-16 00:00).
# Exit code 1 when a pair run failed or left no model; marker temp/e317/E317_DONE only on exit 0 (the chain of E318 waits for it). Log temp/e317/chain.log. Never edit this script while it runs.
cd "$(dirname "$0")/../.."
renice -n 10 -p $$ > /dev/null 2>&1                                                                  # the post-processing too runs niced (D98: the machine stays usable)
mkdir -p temp/e317
L=temp/e317/chain.log; LOGD=temp/e317
. temp/e330/recipe.sh
. temp/e330/chain_lib.sh
CONTROL_PREFIX=e312_p_s                                                                              # D96: the control of this card stays E312 (recipe.sh may be re-based on E324 s for the removal arms: not for this card)
READOUT_EXTRA="--snap --snap-nogeom"                                                                 # post_pair adds the snapped read-out with the keypoint geometry (the end-geometry claim)
host_ok() { R=8; Lm=8; [ "$(date +%s)" -lt "$(date -d "2026-11-16 00:00" +%s)" ] && { R=6; Lm=10; }; [ "$(free -g | awk '/^Mem:/{print $7}')" -ge $R ] && [ "$(awk '{print int($1)}' /proc/loadavg)" -lt $Lm ]; }     # D98: free RAM >= 8 GB, load < 8; loosened to 6 GB / 10 only until Monday 2026-11-16 00:00 (Mirko: not permanent)
wait_marker "$1"
RC=0
for S in 0 1; do
  until can_start && host_ok; do echo "e317 seed $S waits for the window / the host $(date)" >> $L; sleep 300; done
  bash temp/e317/start_pairs.sh $S >> $LOGD/lane$S.log 2>&1; X=$?; echo "e317 seed $S exit $X $(date)" >> $L
  { [ $X = 0 ] && [ -f runs/e317_g_s$S/model.pt ]; } || { RC=1; echo "e317 seed $S FAILED (exit $X or no model) $(date)" >> $L; }
  post_pair e317_g_s$S $S
done
for S in 0 1; do                                                                                     # E312 unsnapped WITH counts: a tagged card, nothing stored is overwritten (never twice: a tagged card is not overwritten)
  [ -f runs/e312_p_s$S/ends_card_cnt_thr0.1.json ] || bash tools/env/_py14.sh jends_eval.py score --run e312_p_s$S --seed $S --thr 0.1 --tag cnt >> $LOGD/rescore_e312_s$S.log 2>&1
done
for S in 0 1; do
  R=e317_g_s$S
  [ -f runs/$R/model.pt ] || continue
  bash tools/env/_py14.sh jdiag.py kperr $R e312_p_s$S >> $LOGD/kperr_s$S.log 2>&1                    # the VAL claim on the end error (strict one-to-one, p >= 0.5)
  (cd temp/e330 && bash ../../tools/env/_py14.sh e330/kill_check.py ${R}_maps $S 0.08 >> ../e317/flag_maps_s$S.log 2>&1)    # REPORT-ONLY (its exit code is ignored): the maps keypoint F1 against E307, flagged below -0.08
  bash tools/env/_py14.sh jdiag.py boot $R ends_card_thr0.1 e312_p_s$S ends_card_cnt_thr0.1 >> $LOGD/boot_vs_e312_unsnap_s$S.log 2>&1
  bash tools/env/_py14.sh jdiag.py boot $R ends_card_snap_thr0.1 e312_p_s$S ends_card_snap_thr0.1 >> $LOGD/boot_vs_e312_snap_s$S.log 2>&1
  bash tools/env/_py14.sh jdiag.py boot $R ends_card_snap_nogeom_thr0.1 e312_p_s$S ends_card_snap_nogeom_thr0.1 >> $LOGD/boot_vs_e312_nogeom_s$S.log 2>&1
  for C in snap snap_nogeom; do                                                                      # context only (D94): E324 s on its own keypoints, other steps (6000) and the attachment
    [ -f runs/e324_s_s$S/ends_card_${C}_thr0.1.counts.pkl ] && bash tools/env/_py14.sh jdiag.py boot $R ends_card_${C}_thr0.1 e324_s_s$S ends_card_${C}_thr0.1 >> $LOGD/boot_vs_e324s_${C}_s$S.log 2>&1
  done
done
[ $RC = 0 ] && touch temp/e317/E317_DONE
echo "done RC=$RC $(date)" >> $L
exit $RC
