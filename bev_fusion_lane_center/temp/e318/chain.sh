#!/bin/bash
# E318 chain (D96 / D98): the pair stage with the position scale (temp/e318/start_pairs.sh, arm u) for the two seeds ONE AFTER THE OTHER (one GPU job at a time, nice), per seed the post-processing of temp/e330/chain_lib.sh
# (pair PR against the CONTROL E317.g of the same seed), then the scale check on the dump (temp/e318/sigma_check.py: Spearman, fifths, clamp share) and the paired scene bootstraps against the control E317.g and,
# as context only (D94), against E312 and E324 s. The control is E317.g (the card): NO c arm is run; when runs/e317_g_s<S> is missing that seed is not started and the chain exits 1 (the fallback is the c arm
# `temp/e318/start_pairs.sh <seed> c` after the hold of E318.c_s<seed> is cleared: the lead's decision).
# usage: chain.sh [marker file to wait for | none]; the default waits for the end of the E317 chain (its log says "done RC="), `none` = do not wait. Starts only when chain_lib.sh can_start and host_ok (free RAM >= 8 GB, load < 8; 6 / 10 until Mon 2026-11-16 00:00).
# Exit code 1 when a pair run failed or left no model or a control is missing; marker temp/e318/E318_DONE only on exit 0. Log temp/e318/chain.log. Never edit this script while it runs.
cd "$(dirname "$0")/../.."
renice -n 10 -p $$ > /dev/null 2>&1
mkdir -p temp/e318
L=temp/e318/chain.log; LOGD=temp/e318
. temp/e330/recipe.sh
. temp/e330/chain_lib.sh
CONTROL_PREFIX=e317_g_s                                                                              # the control of E318 is E317.g of the same seed (the card); post_pair's pair PR reads against it
READOUT_EXTRA="--snap --snap-nogeom"                                                                 # as E317: the snapped read-out with the keypoint geometry
host_ok() { R=8; Lm=8; [ "$(date +%s)" -lt "$(date -d "2026-11-16 00:00" +%s)" ] && { R=6; Lm=10; }; [ "$(free -g | awk '/^Mem:/{print $7}')" -ge $R ] && [ "$(awk '{print int($1)}' /proc/loadavg)" -lt $Lm ]; }     # D98: free RAM >= 8 GB, load < 8; loosened to 6 GB / 10 only until Monday 2026-11-16 00:00 (Mirko: not permanent)
# wait for the END of the E317 chain (its log says "done RC="), not for E317_DONE: a failed E317 seed must not block the good seed (the per-seed control check below skips only the seed whose control has no model)
if [ "${1-}" != none ]; then until grep -q "^done RC=" temp/e317/chain.log 2>/dev/null; do sleep 60; done; fi
RC=0
for S in 0 1; do
  [ -f runs/e317_g_s$S/model.pt ] || { RC=1; echo "e318 seed $S: the control runs/e317_g_s$S has no model: seed not started (fallback: the c arm, the lead's decision) $(date)" >> $L; continue; }
  until can_start && host_ok; do echo "e318 seed $S waits for the window / the host $(date)" >> $L; sleep 300; done
  bash temp/e318/start_pairs.sh $S >> $LOGD/lane$S.log 2>&1; X=$?; echo "e318 seed $S exit $X $(date)" >> $L
  { [ $X = 0 ] && [ -f runs/e318_u_s$S/model.pt ]; } || { RC=1; echo "e318 seed $S FAILED (exit $X or no model) $(date)" >> $L; }
  post_pair e318_u_s$S $S
  [ -f runs/e318_u_s$S/ends_val.pkl ] && bash tools/env/_py14.sh sigma_check.py e318_u_s$S >> $LOGD/sigma_check_s$S.log 2>&1      # CPU: the GT cache of temp/e318/gt_val.pkl is built on the first call (5 - 8 minutes; it exists after the self-test of 2026-11-14)
done
for S in 0 1; do
  R=e318_u_s$S
  [ -f runs/$R/model.pt ] || continue
  bash tools/env/_py14.sh jdiag.py kperr $R e317_g_s$S e312_p_s$S >> $LOGD/kperr_s$S.log 2>&1
  (cd temp/e330 && bash ../../tools/env/_py14.sh e330/kill_check.py ${R}_maps $S 0.08 >> ../e318/flag_maps_s$S.log 2>&1)    # REPORT-ONLY (exit code ignored): the maps keypoint F1 against E307, flagged below -0.08
  for C in snap snap_nogeom; do                                                                      # against the control E317.g (the rule's comparison), then against E312 (E324 s: context only)
    for B in e317_g_s e312_p_s e324_s_s; do
      [ -f runs/${B}$S/ends_card_${C}_thr0.1.counts.pkl ] && bash tools/env/_py14.sh jdiag.py boot $R ends_card_${C}_thr0.1 ${B}$S ends_card_${C}_thr0.1 >> $LOGD/boot_${C}_vs_${B}${S}.log 2>&1
    done
  done
  [ -f runs/e317_g_s$S/ends_card_thr0.1.counts.pkl ] && bash tools/env/_py14.sh jdiag.py boot $R ends_card_thr0.1 e317_g_s$S ends_card_thr0.1 >> $LOGD/boot_unsnap_vs_e317_g_s$S.log 2>&1
done
[ $RC = 0 ] && touch temp/e318/E318_DONE
echo "done RC=$RC $(date)" >> $L
exit $RC
