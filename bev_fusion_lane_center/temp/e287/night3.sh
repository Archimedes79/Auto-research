#!/bin/bash
# Chain 3 (after the E287 runs): the JOINT maps arm jj of E286 (pyramid trained together) with the context depth of the best arm, its score, then the pair head q2 (= p2 on the jj maps), dump + score.
# Log temp/e287/night3.log; flags JJ_DONE, Q2_DONE. Starts runs only inside the GPU window.
cd "$(dirname "$0")/../.."
L=temp/e287/night3.log
echo "waiting for temp/e287/LANES_DONE $(date)" >> $L
until [ -f temp/e287/LANES_DONE ]; do sleep 60; done
in_window() { H=$(date +%-H); [ "$H" -ge 22 ] || [ "$H" -lt 7 ] || { [ "$H" -ge 8 ] && [ "$H" -lt 17 ]; }; }
BEST=$(bash tools/env/_py14.sh temp/e287/pick_best.py 2>>$L | tail -1)
echo "E286 best arm: $BEST $(date)" >> $L
if ! in_window; then echo "outside the GPU window $(date), stopping" >> $L; exit 0; fi
export JJ_CTX=$(case $BEST in j0) echo none;; j1) echo dil;; j2) echo attn;; esac)
lane() {
  S=$1
  in_window || { echo "lane $S: window over $(date)" >> $L; return; }
  bash temp/e286/start_jmaps.sh jj $S >> temp/e287/jj_lane$S.log 2>&1
  echo "jj lane $S exit $? $(date)" >> $L
  [ -f runs/e286_jj_s$S/model.pt ] || return
  bash tools/env/_pyr.sh jmaps_eval.py score --run e286_jj_s$S >> temp/e287/jj_score_s$S.log 2>&1
  in_window || { echo "lane $S: window over before q2 $(date)" >> $L; return; }
  bash temp/e287/start_pairs.sh q2 $S jj >> temp/e287/q2_lane$S.log 2>&1
  echo "q2 lane $S exit $? $(date)" >> $L
  if [ -f runs/e287_q2_s$S/model.pt ]; then
    bash tools/env/_pyr.sh jends_eval.py dump --run e287_q2_s$S >> temp/e287/dump_q2_s$S.log 2>&1
    bash tools/env/_py14.sh jends_eval.py score --run e287_q2_s$S --seed $S >> temp/e287/score_q2_s$S.log 2>&1
    echo "q2 scored $S exit $? $(date)" >> $L
  fi
}
lane 0 &
lane 1 &
wait
touch temp/e287/JJ_DONE
echo "done $(date)" >> $L
