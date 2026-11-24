#!/bin/bash
# Day chain 20 (2026-11-14, lane experiments chat; D100 = the owner: "vergiss die Wochendregel, lass laufen": the weekend rule D98 is void).
# E317 -> E318 (arm u, control E317.g of the same seed) -> E319 in TWO LANES, one per seed, that run in parallel; every GPU job goes through run.py, which hands out its two slots first come first served
# (E341.p_s1, day14, day16 and the BEV chat's jobs stand in the same line). NO host gate, NO renice, NO nice (start scripts of 2026-11-14 changed). Stays: the window rule of chain_lib.sh (can_start: nights,
# weekends, Mon-Fri 08:30 - 15:30), the launcher's slots and memory limits, the holds (E318.c_s0 / E318.c_s1 stay held: the c arm is the fallback only, the control is E317.g).
# The experiment logs and markers are those of the three chain scripts (temp/e317|e318|e319/chain.log; marker E317_DONE etc. only without a failure; "done RC=" at the end), the commands are theirs (read-outs of the cards unchanged).
# Failures: a seed whose run failed or left no model is noted (temp/day20/fail_<start>/), the other lane goes on; E318 seed S starts only when runs/e317_g_s<S> has a model.
# Dry run (no run, no marker, nothing outside temp/day20/dry): DAY20_DRY=1 [DAY20_GATE=0|1|2] bash temp/day20/day20.sh. Never edit this script while it runs. Log temp/day20/day20.log, marker temp/day20/DAY20_DONE.
cd "$(dirname "$0")/../.."
D=temp/day20
if [ -n "$DAY20_DRY" ]; then D=temp/day20/dry_$(date +%H%M%S); T317=$D/e317; T318=$D/e318; T319=$D/e319; LOCK=/tmp/day20_dry.lock; else T317=temp/e317; T318=temp/e318; T319=temp/e319; LOCK=/tmp/day20.lock; fi
L20=$D/day20.log
mkdir -p $D $T317 $T318 $T319
exec 9>$LOCK
flock -n 9 || { echo "day20 already runs $(date)" >> $L20; exit 0; }
[ -f $D/DAY20_DONE ] && { echo "day20 already done $(date)" >> $L20; exit 0; }
F=$D/fail_$(date +%s); mkdir -p $F
. temp/e330/recipe.sh
. temp/e330/chain_lib.sh                                  # can_start, wait_start, post_pair (reads $L $LOGD $READOUT_EXTRA $CONTROL_PREFIX: set per experiment in the lane)
# gate <seed>: as temp/e319/chain.sh (the maps GAIN over E307: 0 = stage 2 runs, 1 = it does not, 2 = the check failed)
gate() {
  local S=$1 OUT X
  OUT=$(cd temp/e319 && bash ../../tools/env/_py14.sh e319/kill_check.py $S $GATE_RUNS 2>&1); X=$?
  echo "$OUT" >> temp/e319/kill_s$S.log
  if [ $X = 0 ] && echo "$OUT" | grep -q "^GATE met"; then return 0; fi
  if [ $X = 1 ] && echo "$OUT" | grep -q "^GATE not met"; then return 1; fi
  return 2
}
if [ -n "$DAY20_DRY" ]; then
  X() { echo "DRY: $*"; sleep 0.$((RANDOM % 9 + 1)); }; has() { true; }; can_start() { true; }; gate() { return ${DAY20_GATE:-0}; }; post_pair() { echo "DRY post_pair $* ($L $LOGD $CONTROL_PREFIX '$READOUT_EXTRA')" >> $L; }
else
  X() { "$@"; }; has() { [ -f "$1" ]; }
fi
fail() { echo "$1 $(date)" >> $2; touch $F/$3; }       # fail <text> <log> <marker name>

e317_seed() {   # the pair stage with the geometry loss, post-processing, the reads of the card
  local S=$1 XC R=e317_g_s$1 LA=$PWD/$T317
  L=$T317/chain.log; LOGD=$T317; CONTROL_PREFIX=e312_p_s; READOUT_EXTRA="--snap --snap-nogeom"
  wait_start
  X bash temp/e317/start_pairs.sh $S >> $LOGD/lane$S.log 2>&1; XC=$?; echo "e317 seed $S exit $XC $(date)" >> $L
  { [ $XC = 0 ] && has runs/$R/model.pt; } || fail "e317 seed $S FAILED (exit $XC or no model)" $L e317_s$S
  post_pair $R $S
  has runs/e312_p_s$S/ends_card_cnt_thr0.1.json || X bash tools/env/_py14.sh jends_eval.py score --run e312_p_s$S --seed $S --thr 0.1 --tag cnt >> $LOGD/rescore_e312_s$S.log 2>&1    # E312 unsnapped WITH counts (a tagged card is never overwritten)
  has runs/$R/model.pt || return 0
  X bash tools/env/_py14.sh jdiag.py kperr $R e312_p_s$S >> $LOGD/kperr_s$S.log 2>&1                    # the VAL claim on the end error (strict one-to-one, p >= 0.5)
  (cd temp/e330 && X bash ../../tools/env/_py14.sh e330/kill_check.py ${R}_maps $S 0.08 >> $LA/flag_maps_s$S.log 2>&1)    # REPORT-ONLY (exit code ignored): the maps keypoint F1 against E307
  X bash tools/env/_py14.sh jdiag.py boot $R ends_card_thr0.1 e312_p_s$S ends_card_cnt_thr0.1 >> $LOGD/boot_vs_e312_unsnap_s$S.log 2>&1
  X bash tools/env/_py14.sh jdiag.py boot $R ends_card_snap_thr0.1 e312_p_s$S ends_card_snap_thr0.1 >> $LOGD/boot_vs_e312_snap_s$S.log 2>&1
  X bash tools/env/_py14.sh jdiag.py boot $R ends_card_snap_nogeom_thr0.1 e312_p_s$S ends_card_snap_nogeom_thr0.1 >> $LOGD/boot_vs_e312_nogeom_s$S.log 2>&1
  for C in snap snap_nogeom; do                                                                          # context only (D94): E324 s on its own keypoints, other steps (6000) and the attachment
    has runs/e324_s_s$S/ends_card_${C}_thr0.1.counts.pkl && X bash tools/env/_py14.sh jdiag.py boot $R ends_card_${C}_thr0.1 e324_s_s$S ends_card_${C}_thr0.1 >> $LOGD/boot_vs_e324s_${C}_s$S.log 2>&1
  done
}

e318_seed() {   # arm u (position scale); the control is E317.g of the same seed, NO c arm
  local S=$1 XC R=e318_u_s$1 LA=$PWD/$T318 C B
  L=$T318/chain.log; LOGD=$T318; CONTROL_PREFIX=e317_g_s; READOUT_EXTRA="--snap --snap-nogeom"
  has runs/e317_g_s$S/model.pt || { fail "e318 seed $S: the control runs/e317_g_s$S has no model: seed not started (fallback: the c arm, the lead's decision)" $L e318_s$S; return 0; }
  wait_start
  X bash temp/e318/start_pairs.sh $S >> $LOGD/lane$S.log 2>&1; XC=$?; echo "e318 seed $S exit $XC $(date)" >> $L
  { [ $XC = 0 ] && has runs/$R/model.pt; } || fail "e318 seed $S FAILED (exit $XC or no model)" $L e318_s$S
  post_pair $R $S
  has runs/$R/ends_val.pkl && X bash tools/env/_py14.sh sigma_check.py $R >> $LOGD/sigma_check_s$S.log 2>&1      # CPU: the GT cache of temp/e318/gt_val.pkl exists since the self-test of 2026-11-14
  has runs/$R/model.pt || return 0
  X bash tools/env/_py14.sh jdiag.py kperr $R e317_g_s$S e312_p_s$S >> $LOGD/kperr_s$S.log 2>&1
  (cd temp/e330 && X bash ../../tools/env/_py14.sh e330/kill_check.py ${R}_maps $S 0.08 >> $LA/flag_maps_s$S.log 2>&1)    # REPORT-ONLY (exit code ignored)
  for C in snap snap_nogeom; do                                                                          # against the control E317.g (the rule's comparison), then E312 (E324 s: context only)
    for B in e317_g_s e312_p_s e324_s_s; do
      has runs/${B}$S/ends_card_${C}_thr0.1.counts.pkl && X bash tools/env/_py14.sh jdiag.py boot $R ends_card_${C}_thr0.1 ${B}$S ends_card_${C}_thr0.1 >> $LOGD/boot_${C}_vs_${B}${S}.log 2>&1
    done
  done
  has runs/e317_g_s$S/ends_card_thr0.1.counts.pkl && X bash tools/env/_py14.sh jdiag.py boot $R ends_card_thr0.1 e317_g_s$S ends_card_thr0.1 >> $LOGD/boot_unsnap_vs_e317_g_s$S.log 2>&1
}

e319_seed() {   # maps with the heading-guided exchange -> maps score -> the GATE (a maps gain over E307) -> the E312 pair recipe on those maps -> post_pair -> the reads of the card
  local S=$1 XC G RAN=0 R=e319_p_s$1
  L=$T319/chain.log; LOGD=$T319; CONTROL_PREFIX=e312_p_s; READOUT_EXTRA=""
  wait_start
  X bash temp/e319/start_maps.sh $S >> $LOGD/lane$S.log 2>&1; XC=$?; echo "e319 seed $S maps exit $XC $(date)" >> $L
  if ! { [ $XC = 0 ] && has runs/e319_m_s$S/model.pt; }; then fail "e319 seed $S maps FAILED (exit $XC or no model): pair stage not run" $L e319_s$S; return 0; fi
  X bash tools/env/_pyr.sh jmaps_eval.py score --run e319_m_s$S >> $LOGD/score_e319_m_s$S.log 2>&1
  gate $S; G=$?
  case $G in
    0) echo "e319 seed $S gate met (maps gain over E307): pair stage runs $(date)" >> $L
       wait_start
       X bash temp/e319/start_pairs.sh $S >> $LOGD/lane$S.log 2>&1; XC=$?; echo "e319 seed $S pairs exit $XC $(date)" >> $L
       { [ $XC = 0 ] && has runs/$R/model.pt; } || fail "e319 seed $S pairs FAILED (exit $XC or no model)" $L e319_s$S
       post_pair $R $S; RAN=1 ;;
    1) echo "e319 seed $S gate NOT met (no maps gain over E307, or the kill): pair stage not run, see temp/e319/kill_s$S.log $(date)" >> $L ;;
    *) fail "e319 seed $S gate CHECK FAILED (see temp/e319/kill_s$S.log): pair stage not run" $L e319_s$S ;;
  esac
  [ $RAN = 1 ] && has runs/$R/model.pt || return 0
  has runs/e312_p_s$S/ends_card_cnt_thr0.1.json || X bash tools/env/_py14.sh jends_eval.py score --run e312_p_s$S --seed $S --thr 0.1 --tag cnt >> $LOGD/rescore_e312_s$S.log 2>&1
  X bash tools/env/_py14.sh jdiag.py boot $R ends_card_thr0.1 e312_p_s$S ends_card_cnt_thr0.1 >> $LOGD/boot_vs_e312_unsnap_s$S.log 2>&1
  X bash tools/env/_py14.sh jdiag.py boot $R ends_card_snap_thr0.1 e312_p_s$S ends_card_snap_thr0.1 >> $LOGD/boot_vs_e312_snap_s$S.log 2>&1
  has runs/e324_s_s$S/ends_card_snap_thr0.1.counts.pkl && X bash tools/env/_py14.sh jdiag.py boot $R ends_card_snap_thr0.1 e324_s_s$S ends_card_snap_thr0.1 >> $LOGD/boot_vs_e324s_snap_s$S.log 2>&1   # context only (D94)
}

lane() { e317_seed $1; e318_seed $1; e319_seed $1; echo "lane seed $1 finished $(date)" >> $L20; }

echo "day20 starts $(date) (dry: ${DAY20_DRY:-no})" >> $L20
lane 0 & P0=$!
lane 1 & P1=$!
wait $P0 $P1
X bash tools/env/_py14.sh jdiag.py maps e319_m e307_n >> $T319/maps_vs_e307.log 2>&1                  # keypoint F1, zone IoU, crowded / isolated / far entry recall of both seeds against E307
for E in 317 318 319; do
  eval "T=\$T$E"; RC=0
  ls $F/e${E}_s* > /dev/null 2>&1 && RC=1
  [ $RC = 0 ] && touch $T/E${E}_DONE
  echo "done RC=$RC $(date)" >> $T/chain.log
done
touch $D/DAY20_DONE
echo "done $(date)" >> $L20
