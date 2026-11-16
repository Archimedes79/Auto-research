#!/bin/bash
# Day chain 3 (replaces day.sh + day2b.sh, which waited behind each other and scored everything at the end): per lane (seed 0 / seed 1) in the order of what the pair-head finding of E308 asks for:
# E305 (maps + pair head together) -> E302 (capacitated pair head) -> E307 (narrow end heat) -> E298 (driven weight) -> E306 (exact raster) -> E303 (handle scales) -> E304 (far-range maps).
# Every run is dumped / scored in the background beside the next training (GPU dump is short, the scores are CPU, 1 thread). Runs start only inside the GPU window (nights 22 - 07, weekdays 08:30 - 16:30 for a START so that a run ends before 17:30).
# Log temp/day3/day3.log; flag temp/day3/DAY3_DONE. Never edit this script while it runs.
cd "$(dirname "$0")/../.."
L=temp/day3/day3.log
mkdir -p temp/day3
can_start() { H=$(date +%-H); M=$(date +%-M); [ "$H" -ge 22 ] || [ "$H" -lt 7 ] || { { [ "$H" -gt 8 ] || { [ "$H" -eq 8 ] && [ "$M" -ge 30 ]; }; } && { [ "$H" -lt 16 ] || { [ "$H" -eq 16 ] && [ "$M" -lt 30 ]; }; }; }; }
wait_start() { until can_start; do sleep 60; done; }
BEST=$(bash tools/env/_py14.sh temp/e287/pick_best.py 2>>$L | tail -1)
CTX=$(case $BEST in j0) echo none;; j1) echo dil;; j2) echo attn;; esac)
echo "day3 start: E286 best arm $BEST ($CTX) $(date)" >> $L
post_pair() {   # dump + score + PR of a pair-head run
  R=$1; S=$2
  [ -f runs/$R/model.pt ] || { echo "no model of $R $(date)" >> $L; return; }
  bash tools/env/_pyr.sh jends_eval.py dump --run $R >> temp/day3/dump_$R.log 2>&1
  bash tools/env/_py14.sh jends_eval.py score --run $R --seed $S >> temp/day3/score_$R.log 2>&1
  bash tools/env/_py14.sh jpair_pr.py score --run $R --control e287_p2_s$S >> temp/day3/pr_$R.log 2>&1
  echo "scored $R $(date)" >> $L
}
post_maps() {   # score of a maps-head run (extra args: --raster exact for a control)
  R=$1; shift
  [ -f runs/$R/model.pt ] || { echo "no model of $R $(date)" >> $L; return; }
  bash tools/env/_pyr.sh jmaps_eval.py score --run $R "$@" >> temp/day3/score_$R.log 2>&1
  echo "scored $R $* $(date)" >> $L
}
lane() {
  S=$1
  wait_start; bash temp/e305/start_e2e.sh e $S $BEST >> temp/day3/lane$S.log 2>&1; echo "lane $S E305 exit $? $(date)" >> $L
  ( post_pair e305_e_s$S $S; post_maps e305_e_s${S}_maps ) &
  wait_start; bash temp/e302/start_cap.sh $S $BEST >> temp/day3/lane$S.log 2>&1; echo "lane $S E302 exit $? $(date)" >> $L
  post_pair e302_k_s$S $S &
  wait_start; bash temp/e307/start_narrow.sh $S >> temp/day3/lane$S.log 2>&1; echo "lane $S E307 exit $? $(date)" >> $L
  post_maps e307_n_s$S &
  wait_start; bash temp/e298/start_driven.sh $S $BEST >> temp/day3/lane$S.log 2>&1; echo "lane $S E298 exit $? $(date)" >> $L
  post_pair e298_w_s$S $S &
  wait_start; bash temp/e306/start_exact.sh $S >> temp/day3/lane$S.log 2>&1; echo "lane $S E306 exit $? $(date)" >> $L
  ( post_maps e306_x_s$S; post_maps e286_j1_s$S --raster exact ) &
  wait_start; bash temp/e303/start_scales.sh $S $BEST >> temp/day3/lane$S.log 2>&1; echo "lane $S E303 exit $? $(date)" >> $L
  ( post_pair e303_s_s$S $S
    if [ -f runs/e303_s_s$S/model.pt ]; then
      mkdir -p runs/e303_chord_s$S && cp runs/e303_s_s$S/ends_val.pkl runs/e303_s_s$S/model.pt runs/e303_chord_s$S/ 2>/dev/null
      bash tools/env/_py14.sh jends_eval.py score --run e303_chord_s$S --seed $S --chord >> temp/day3/score_e303_chord_s$S.log 2>&1
    fi ) &
  wait_start; bash temp/e304/start_far.sh $S $CTX >> temp/day3/lane$S.log 2>&1; echo "lane $S E304 exit $? $(date)" >> $L
  post_maps e304_f_s$S &
  wait
}
lane 0 &
lane 1 &
wait
touch temp/day3/DAY3_DONE
echo "done $(date)" >> $L
