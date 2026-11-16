#!/bin/bash
# Day chain 24 (2026-11-15, lane experiments chat; card E348): the E324 s recipe on the maps head without the 0.5 m paint branch, TWO LANES (one per seed, parallel where run.py has slots), then the removal rule.
# Per lane: wait_start -> temp/e348/start_pairs.sh <seed> -> post_pair of temp/e330/chain_lib.sh (dump through the GPU gate, unsnapped, snapped and PRIMARY scores, plain card, pair PR against E324 s, maneuver coverage, maps score)
# -> by hand: attach-only score, snap-nogeom (context), kperr against E324 s, the maps flag (report only), the paired scene bootstraps against E324 s on the unsnapped, snapped, PRIMARY, attach and snap-nogeom cards.
# After both lanes: `jrule_removal.py --arm e348_w_s --control e324_s_s --readouts snap primary attach` -> temp/e348/rule.log (exit 0 = ADOPT, 1 = not adopted, 2 = a card is missing); the label ADOPT-FIRM / ADOPT-POINT is read from the bootstraps by hand.
# Stage 0 (the unsnapped and attach-only cards of the CONTROL E324 s) was done by temp/e348/stage0.sh. Never edit while it runs. Run-once guard, log temp/e348/chain.log, marker temp/day24/DAY24_DONE.
# Dry run: DAY24_DRY=1 bash temp/day24/day24.sh (nothing is run, everything under temp/day24/dry_*).
cd "$(dirname "$0")/../.."
D=temp/day24
[ -n "$DAY24_DRY" ] && D=temp/day24/dry_$(date +%H%M%S)
T=temp/e348
[ -n "$DAY24_DRY" ] && T=$D/e348
mkdir -p $D $T
L22=$D/day24.log
exec 9>/tmp/day24${DAY24_DRY:+_dry}.lock
flock -n 9 || { echo "day24 already runs $(date)" >> $L22; exit 0; }
[ -f $D/DAY24_DONE ] && { echo "day24 already done $(date)" >> $L22; exit 0; }
. temp/e330/recipe.sh
. temp/e330/chain_lib.sh
if [ -n "$DAY24_DRY" ]; then
  X() { echo "DRY: $*"; }; has() { true; }; can_start() { true; }; post_pair() { echo "DRY post_pair $* ($L $LOGD $CONTROL_PREFIX '$READOUT_EXTRA')" >> $L; }
else
  X() { "$@"; }; has() { [ -f "$1" ]; }
fi

lane() {
  local S=$1 XC R=e348_w_s$1 C=e324_s_s$1 LA=$PWD/$T B
  L=$T/chain.log; LOGD=$T; CONTROL_PREFIX=e324_s_s; READOUT_EXTRA="--attach --snap --snap-nogeom"
  wait_start
  X bash temp/e348/start_pairs.sh $S >> $LOGD/lane$S.log 2>&1; XC=$?; echo "e348 seed $S exit $XC $(date)" >> $L
  { [ $XC = 0 ] && has runs/$R/model.pt; } || { echo "e348 seed $S FAILED (exit $XC or no model) $(date)" >> $L; return 1; }
  post_pair $R $S
  has runs/$R/model.pt || return 1
  X bash tools/env/_py14.sh jends_eval.py score --run $R --seed $S --attach --thr 0.1 >> $LOGD/score_attach_$R.log 2>&1
  X bash tools/env/_py14.sh jends_eval.py score --run $R --seed $S --snap --snap-nogeom --thr 0.1 >> $LOGD/score_nogeom_$R.log 2>&1
  X bash tools/env/_py14.sh jdiag.py kperr $R $C >> $LOGD/kperr_s$S.log 2>&1
  (cd temp/e330 && X bash ../../tools/env/_py14.sh e330/kill_check.py ${R}_maps $S 0.08 >> $LA/flag_maps_s$S.log 2>&1)       # REPORT-ONLY (exit code ignored)
  for B in thr0.1 snap_thr0.1 snap_nogeom_attach_thr0.1 attach_thr0.1 snap_nogeom_thr0.1; do                              # arm card against the control's card of the same name
    X bash tools/env/_py14.sh jdiag.py boot $R ends_card_$B $C ends_card_$B >> $LOGD/boot_${B}_s$S.log 2>&1
  done
  echo "e348 seed $S post-processing done $(date)" >> $L
}

echo "day24 starts $(date) (dry: ${DAY24_DRY:-no})" >> $L22
lane 0 & P0=$!
lane 1 & P1=$!
R0=0; wait $P0 || R0=1; R1=0; wait $P1 || R1=1
X bash tools/env/_py14.sh jrule_removal.py --arm e348_w_s --control e324_s_s --readouts snap primary attach >> $T/rule.log 2>&1
echo "e348 rule exit $? (0 ADOPT, 1 not adopted, 2 a card missing); lanes $R0 $R1 $(date)" >> $T/chain.log
touch $D/DAY24_DONE
echo "done $(date)" >> $L22
