#!/bin/bash
# E319 chain (D96 / D98): the two seeds ONE AFTER THE OTHER (one GPU job at a time, nice). Per seed: maps with dirx (start_maps.sh) -> maps score -> the GATE of stage 2 (kill_check.py: a maps GAIN over E307, card section 0)
# -> the E312 pair recipe on those maps (start_pairs.sh) -> post_pair of temp/e330/chain_lib.sh (dump, unsnapped / snapped scores, the plain card, pair PR and maneuver coverage against E312, the joint maps' own score).
# After both seeds: the maps table against E307, E312's unsnapped card WITH counts (tag cnt) and the paired scene bootstraps against E312 (and, as context only (D94), against E324 s) for every seed whose pair stage ran.
# usage: chain.sh [marker file to wait for]. Starts only when chain_lib.sh can_start (nights, Mon-Fri 08:30-15:30, weekend days since D98) and host_ok (free RAM >= 8 GB, load < 8). Never edit this script while it runs.
# Gate: exit 0 of kill_check.py + the line `GATE met` = stage 2 runs; exit 1 + `GATE not met` = it does not (a result: the log line says so, the chain stays OK); anything else (exit 2, a crash, a missing token) = the check FAILED:
# stage 2 does not run on that seed and the chain exits 1. Exit code 1 also when a maps or pair run failed or left no model. Marker temp/e319/E319_DONE only on exit 0. Log temp/e319/chain.log.
# (The main code runs only when the file is executed, not when it is sourced: the gate function is tested by sourcing.)
cd "$(dirname "$0")/../.."
mkdir -p temp/e319
L=temp/e319/chain.log; LOGD=temp/e319
. temp/e330/recipe.sh
. temp/e330/chain_lib.sh
CONTROL_PREFIX=e312_p_s                                                                              # the control of this card is E312 of the same seed (D96); recipe.sh may be re-based on E324 s for the removal arms: not for this card
READOUT_EXTRA=""
host_ok() { [ "$(free -g | awk '/^Mem:/{print $7}')" -ge 8 ] && [ "$(awk '{print int($1)}' /proc/loadavg)" -lt 8 ]; }     # D98: free RAM >= 8 GB, load < 8
# gate <seed>: the check output goes to temp/e319/kill_s<seed>.log; return 0 = stage 2 runs, 1 = it does not (no maps gain / the kill), 2 = the check failed. GATE_RUNS (empty in the chain) = '<maps run> <reference run>' for the test of this function
gate() {
  local S=$1 OUT X
  OUT=$(cd temp/e319 && bash ../../tools/env/_py14.sh e319/kill_check.py $S $GATE_RUNS 2>&1); X=$?
  echo "$OUT" >> temp/e319/kill_s$S.log
  if [ $X = 0 ] && echo "$OUT" | grep -q "^GATE met"; then return 0; fi
  if [ $X = 1 ] && echo "$OUT" | grep -q "^GATE not met"; then return 1; fi
  return 2
}
main() {
  renice -n 10 -p $$ > /dev/null 2>&1                                                                # the post-processing too runs niced (D98: the machine stays usable)
  wait_marker "$1"
  RC=0
  for S in 0 1; do
    until can_start && host_ok; do echo "e319 seed $S maps waits for the window / the host $(date)" >> $L; sleep 300; done
    bash temp/e319/start_maps.sh $S >> $LOGD/lane$S.log 2>&1; X=$?; echo "e319 seed $S maps exit $X $(date)" >> $L
    if ! { [ $X = 0 ] && [ -f runs/e319_m_s$S/model.pt ]; }; then RC=1; echo "e319 seed $S maps FAILED (exit $X or no model): pair stage not run $(date)" >> $L; continue; fi
    bash tools/env/_pyr.sh jmaps_eval.py score --run e319_m_s$S >> $LOGD/score_e319_m_s$S.log 2>&1
    gate $S; G=$?
    case $G in
      0) echo "e319 seed $S gate met (maps gain over E307): pair stage runs $(date)" >> $L
         until can_start && host_ok; do echo "e319 seed $S pairs waits for the window / the host $(date)" >> $L; sleep 300; done
         bash temp/e319/start_pairs.sh $S >> $LOGD/lane$S.log 2>&1; X=$?; echo "e319 seed $S pairs exit $X $(date)" >> $L
         { [ $X = 0 ] && [ -f runs/e319_p_s$S/model.pt ]; } || { RC=1; echo "e319 seed $S pairs FAILED (exit $X or no model) $(date)" >> $L; }
         post_pair e319_p_s$S $S ;;
      1) echo "e319 seed $S gate NOT met (no maps gain over E307, or the kill): pair stage not run, see temp/e319/kill_s$S.log $(date)" >> $L ;;
      *) RC=1; echo "e319 seed $S gate CHECK FAILED (see temp/e319/kill_s$S.log): pair stage not run $(date)" >> $L ;;
    esac
  done
  bash tools/env/_py14.sh jdiag.py maps e319_m e307_n >> $LOGD/maps_vs_e307.log 2>&1               # keypoint F1, zone IoU, crowded / isolated / far entry recall of both seeds against E307
  for S in 0 1; do
    R=e319_p_s$S
    [ -f runs/$R/model.pt ] || continue
    [ -f runs/e312_p_s$S/ends_card_cnt_thr0.1.json ] || bash tools/env/_py14.sh jends_eval.py score --run e312_p_s$S --seed $S --thr 0.1 --tag cnt >> $LOGD/rescore_e312_s$S.log 2>&1      # E312 unsnapped WITH counts (a tagged card is never overwritten)
    bash tools/env/_py14.sh jdiag.py boot $R ends_card_thr0.1 e312_p_s$S ends_card_cnt_thr0.1 >> $LOGD/boot_vs_e312_unsnap_s$S.log 2>&1
    bash tools/env/_py14.sh jdiag.py boot $R ends_card_snap_thr0.1 e312_p_s$S ends_card_snap_thr0.1 >> $LOGD/boot_vs_e312_snap_s$S.log 2>&1
    [ -f runs/e324_s_s$S/ends_card_snap_thr0.1.counts.pkl ] && bash tools/env/_py14.sh jdiag.py boot $R ends_card_snap_thr0.1 e324_s_s$S ends_card_snap_thr0.1 >> $LOGD/boot_vs_e324s_snap_s$S.log 2>&1   # context only (D94)
  done
  [ $RC = 0 ] && touch temp/e319/E319_DONE
  echo "done RC=$RC $(date)" >> $L
  return $RC
}
if [ "${BASH_SOURCE[0]}" = "$0" ]; then main "$@"; exit $?; fi
