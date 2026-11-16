#!/bin/bash
# E341 chain (D98): the pair stage on the E332 arm-a maps, the two seeds ONE AFTER THE OTHER (by day one GPU job, about 3.0 h wall), the post-processing, the rule and the intervals. Called by temp/day18/day18.sh.
# Exit code 1 when a pair run failed or left no model (a jrule exit 1 is a verdict, not a failure); marker temp/e341/E341_DONE only on exit 0. Never edit this script while it runs.
cd "$(dirname "$0")/../.."
mkdir -p temp/e341
L=temp/e341/chain.log; LOGD=temp/e341
. temp/e330/recipe.sh
. temp/e330/chain_lib.sh
host_ok() { [ "$(free -g | awk '/^Mem:/{print $7}')" -ge 8 ] && [ "$(awk '{print int($1)}' /proc/loadavg)" -lt 8 ]; }     # D98: free RAM >= 8 GB, load < 8
RC=0
for S in 0 1; do
  until can_start && host_ok; do echo "e341 seed $S waits for the window / the host $(date)" >> $L; sleep 300; done
  bash temp/e341/start_pairs.sh $S >> $LOGD/lane$S.log 2>&1; X=$?; echo "e341 seed $S exit $X $(date)" >> $L
  { [ $X = 0 ] && [ -f runs/e341_p_s$S/model.pt ]; } || { RC=1; echo "e341 seed $S FAILED (exit $X or no model) $(date)" >> $L; }
  post_pair e341_p_s$S $S
done
for S in 0 1; do                                                                                       # E312 unsnapped WITH counts: a tagged card, nothing stored is overwritten
  bash tools/env/_py14.sh jends_eval.py score --run e312_p_s$S --seed $S --thr 0.1 --tag cnt >> $LOGD/rescore_e312_s$S.log 2>&1
done
bash tools/env/_py14.sh jrule_removal.py --arm e341_p_s --control $CONTROL_PREFIX --readouts $RULE_READOUTS >> $LOGD/rule_e341.log 2>&1; echo "e341 rule exit $? (1 = not adopted, a verdict) $(date)" >> $L
for S in 0 1; do                                                                                       # intervals: against E332 arm a on both read-outs, against E312 on both
  bash tools/env/_py14.sh jdiag.py boot e341_p_s$S ends_card_thr0.1 e332_pa_s$S ends_card_thr0.1 >> $LOGD/boot_vs_a_unsnap_s$S.log 2>&1
  bash tools/env/_py14.sh jdiag.py boot e341_p_s$S ends_card_snap_thr0.1 e332_pa_s$S ends_card_snap_thr0.1 >> $LOGD/boot_vs_a_snap_s$S.log 2>&1
  bash tools/env/_py14.sh jdiag.py boot e341_p_s$S ends_card_thr0.1 e312_p_s$S ends_card_cnt_thr0.1 >> $LOGD/boot_vs_e312_unsnap_s$S.log 2>&1
  bash tools/env/_py14.sh jdiag.py boot e341_p_s$S ends_card_snap_thr0.1 e312_p_s$S ends_card_snap_thr0.1 >> $LOGD/boot_vs_e312_snap_s$S.log 2>&1
done
[ $RC = 0 ] && touch temp/e341/E341_DONE
echo "done RC=$RC $(date)" >> $L
exit $RC
