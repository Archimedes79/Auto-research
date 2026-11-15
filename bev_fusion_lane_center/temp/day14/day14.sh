#!/bin/bash
# Day chain 14 (2026-11-13, lane chat on the lane lead's order, replaces the waiting day13): a SCHEDULER for the nights Fri / Sat / Sun (GPU windows: starts 22:00 - 05:00 every night and Mon - Fri 08:30 - 15:30; no weekend days).
# It picks, in the order of ORDER, the first item that is READY (temp/<exp>/READY) and ACCEPTED (temp/<exp>/ACCEPTED, for E324 arm s ACCEPTED_s: written after the critic pass) and not DONE (temp/day14/DONE_<item>), waits for the window, runs it (a pair arm = both seeds in parallel in the two GPU slots),
# marks it done and picks again; with nothing ready it waits (60 s) so that an item that becomes ACCEPTED later is picked up by itself (Saturday and Sunday night run without anybody). Order by value under D87 / D88:
# e324s (E320 recipe at 6000 pair steps, 3.2 h) -> e331 (evidence off, resumes the stalled runs) -> e332 (arm a) -> e333 -> e334 (o3). Later items (E292 c2, the E325 head, E324 arm n) are added by a new chain script when their code is tested; never edit this script while it runs.
# Log temp/day14/day14.log, marker temp/day14/DAY14_DONE.
cd "$(dirname "$0")/../.."
L=temp/day14/day14.log; LOGD=temp/day14
mkdir -p $LOGD
echo "day14 starts $(date)" >> $L
. temp/e330/chain_lib.sh          # can_start, wait_start, wait_marker, post_pair, lane; the recipe of the E312 base (the removal arms)
ORDER="e324s e331 e332 e333 e334"
item_ready() {
  case $1 in
    e324s) [ -f temp/e324/READY ] && [ -f temp/e324/ACCEPTED_s ] ;;
    e331) [ -f temp/e331/READY ] && [ -f temp/e331/ACCEPTED ] ;;
    e332) [ -f temp/e332/READY ] && [ -f temp/e332/ACCEPTED ] ;;
    e333) [ -f temp/e333/READY ] && [ -f temp/e333/ACCEPTED ] ;;
    e334) [ -f temp/e334/READY ] && [ -f temp/e334/ACCEPTED ] ;;
    *) return 1 ;;
  esac
}
item_done() { [ -f $LOGD/DONE_$1 ]; }
# a pair-stage removal arm, both seeds in parallel (as day13): <exp lower> <run prefix>
pair_lib() {
  EXP=$1; PFX=$2
  for S in 0 1; do
    ( wait_start; bash temp/$EXP/start_pairs.sh $S >> $LOGD/${EXP}_lane$S.log 2>&1; echo "$EXP seed $S exit $? $(date)" >> $L; post_pair ${PFX}$S $S ) &
  done
  wait
  bash tools/env/_py14.sh jrule_removal.py --arm $PFX --control $CONTROL_PREFIX --readouts $RULE_READOUTS >> $LOGD/rule_$EXP.log 2>&1; echo "$EXP rule exit $? $(date)" >> $L
}
# the post processing of an E320-type run (attached ends): dump, the PRIMARY and the own-keypoint read-outs, pair PR against E312 and E316, kperr, the maps (reported)
post_e324() {
  R=$1; S=$2
  [ -f runs/$R/model.pt ] || { echo "no model of $R $(date)" >> $L; return; }
  flock /tmp/jends_dump.lock bash tools/env/_pyr.sh jends_eval.py dump --run $R >> $LOGD/dump_$R.log 2>&1
  bash tools/env/_py14.sh jends_eval.py score --run $R --seed $S --attach --snap --snap-nogeom --thr 0.1 >> $LOGD/score_primary_$R.log 2>&1
  bash tools/env/_py14.sh jends_eval.py score --run $R --seed $S --snap --thr 0.1 >> $LOGD/score_snap_$R.log 2>&1
  bash tools/env/_py14.sh jends_eval.py score --run $R --seed $S --snap --snap-nogeom --thr 0.1 >> $LOGD/score_nogeom_$R.log 2>&1
  bash tools/env/_py14.sh jpair_pr.py score --run $R --control e312_p_s$S >> $LOGD/pr_e312_$R.log 2>&1
  bash tools/env/_py14.sh jpair_pr.py score --run $R --control e316_p_s$S >> $LOGD/pr_e316_$R.log 2>&1
  bash tools/env/_py14.sh jdiag.py kperr $R e312_p_s$S >> $LOGD/kperr_$R.log 2>&1
  bash tools/env/_pyr.sh jmaps_eval.py score --run ${R}_maps >> $LOGD/maps_$R.log 2>&1
  bash tools/env/_py14.sh jdiag.py boot $R ends_card_snap_nogeom_attach_thr0.1 e312_p_s$S ends_card_snap_thr0.1 >> $LOGD/boot_c_$R.log 2>&1
  bash tools/env/_py14.sh jdiag.py boot $R ends_card_snap_nogeom_attach_thr0.1 e316_p_s$S ends_card_snap_thr0.1 >> $LOGD/boot_l_$R.log 2>&1
  echo "scored $R $(date)" >> $L
}
run_item() {
  case $1 in
    e324s)
      for S in 0 1; do
        ( bash temp/e324/start_pairs.sh $S >> $LOGD/e324_lane$S.log 2>&1; echo "e324 arm s seed $S exit $? $(date)" >> $L; post_e324 e324_s_s$S $S ) &
      done
      wait
      for S in 0 1; do bash tools/env/_py14.sh jdup.py score --seed $S --a-run e324_s_s$S --l-run e316_p_s$S >> $LOGD/jdup_e324_s$S.log 2>&1; done
      ;;
    e331) pair_lib e331 e331_e_s ;;
    e332) bash temp/e332/chain.sh >> $LOGD/e332_chain.out 2>&1; echo "e332 chain exit $? $(date)" >> $L ;;
    e333) bash temp/e333/chain.sh >> $LOGD/e333_chain.out 2>&1; echo "e333 chain exit $? $(date)" >> $L ;;
    e334) pair_lib e334 e334_o_s ;;
  esac
}
while true; do
  wait_start                                   # nothing is picked outside a GPU window: an item that becomes ACCEPTED while we wait is seen at the next pick
  NEXT=""
  for IT in $ORDER; do
    if ! item_done $IT && item_ready $IT; then NEXT=$IT; break; fi
  done
  if [ -z "$NEXT" ]; then
    ALL=1; for IT in $ORDER; do item_done $IT || ALL=0; done
    [ $ALL = 1 ] && break
    sleep 60; continue
  fi
  echo "run $NEXT $(date)" >> $L
  run_item $NEXT
  touch $LOGD/DONE_$NEXT
  echo "done $NEXT $(date)" >> $L
done
touch $LOGD/DAY14_DONE
echo "day14 done $(date)" >> $L
