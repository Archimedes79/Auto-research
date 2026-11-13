#!/bin/bash
# Functions of the simplification chains (sourced by temp/e332/chain.sh, temp/e333/chain.sh): window rule, marker wait, post-processing of a pair run, a maps -> pairs lane.
# Window rule of the lane lead (day12): a START is allowed 22:00 - 05:00 (a 2 h job ends before 07:00) and Mon-Fri 08:30 - 15:30 (ends before 17:30). Never edit a chain while it runs.
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
. temp/e330/recipe.sh
can_start() { H=$(date +%-H); M=$(date +%-M); DOW=$(date +%u); [ "$H" -ge 22 ] || [ "$H" -lt 5 ] || { [ "$DOW" -le 5 ] && { [ "$H" -gt 8 ] || { [ "$H" -eq 8 ] && [ "$M" -ge 30 ]; }; } && { [ "$H" -lt 15 ] || { [ "$H" -eq 15 ] && [ "$M" -lt 30 ]; }; }; }; }
wait_start() { until can_start; do sleep 30; done; }
wait_marker() { [ -n "$1" ] && { echo "waits for $1 $(date)" >> $L; until [ -f "$1" ]; do sleep 30; done; echo "marker found $(date)" >> $L; }; }
# post_pair <pair run> <seed> : dump (one at a time: host RAM), the two read-outs (unsnapped = differentiable, counts; snapped = the rule step), the plain card, pair PR against the control, maneuver coverage, the joint maps' own score; then the rule.
post_pair() {
  R=$1; S=$2
  [ -f runs/$R/model.pt ] || { echo "no model of $R $(date)" >> $L; return; }
  flock /tmp/jends_dump.lock bash tools/env/_pyr.sh jends_eval.py dump --run $R >> $LOGD/dump_$R.log 2>&1
  bash tools/env/_py14.sh jends_eval.py score --run $R --seed $S --thr 0.1 >> $LOGD/score_unsnap_$R.log 2>&1
  bash tools/env/_py14.sh jends_eval.py score --run $R --seed $S --snap --thr 0.1 >> $LOGD/score_snap_$R.log 2>&1
  echo "scored (unsnapped, snapped) $R $(date)" >> $L
  [ -n "$READOUT_EXTRA" ] && bash tools/env/_py14.sh jends_eval.py score --run $R --seed $S $READOUT_EXTRA --thr 0.1 >> $LOGD/score_extra_$R.log 2>&1
  bash tools/env/_py14.sh jends_eval.py score --run $R --seed $S >> $LOGD/score_$R.log 2>&1
  bash tools/env/_py14.sh jpair_pr.py score --run $R --control ${CONTROL_PREFIX}$S >> $LOGD/pr_$R.log 2>&1
  bash tools/env/_py14.sh jmaneuver.py score --run $R --seed $S --variants plain p0.1 p0.1s >> $LOGD/maneuver_$R.log 2>&1
  bash tools/env/_pyr.sh jmaps_eval.py score --run ${R}_maps >> $LOGD/score_${R}_maps.log 2>&1
  echo "scored $R $(date)" >> $L
}
# lane <exp lower e.g. e332> <arm letter> <seed> <maps run> <pair run> <pair extra flags text for the start script is inside temp/<exp>/start_*.sh>: maps stage -> maps score -> kill check -> pair stage -> post_pair
lane() {
  EXP=$1; A=$2; S=$3; MR=$4; PR=$5
  wait_start; bash temp/$EXP/start_maps.sh $S $A >> $LOGD/lane$S.log 2>&1; echo "$EXP arm $A seed $S maps exit $? $(date)" >> $L
  bash tools/env/_pyr.sh jmaps_eval.py score --run $MR >> $LOGD/score_$MR.log 2>&1
  if (cd temp/e330 && bash ../../tools/env/_py14.sh e330/kill_check.py $MR $S 0.05 >> ../$EXP/kill_$MR.log 2>&1); then
    wait_start; bash temp/$EXP/start_pairs.sh $S $A >> $LOGD/lane$S.log 2>&1; echo "$EXP arm $A seed $S pairs exit $? $(date)" >> $L
    post_pair $PR $S
  else
    echo "$EXP arm $A seed $S: kill criterion met, pair stage not run $(date)" >> $L
  fi
}
