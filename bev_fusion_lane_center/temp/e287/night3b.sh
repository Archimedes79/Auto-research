#!/bin/bash
# Chain 3b (replaces night3.sh + night4.sh): after the E287 runs (LANES_DONE) per lane (seed 0 / seed 1): E308 (pair head with the node tokens of B) -> the JOINT maps arm jj of E286 -> its score -> q2 (p2 on the jj maps) -> dump + score -> the E289 line probe. E308's dump / score / PR run beside the next training.
# Log temp/e287/night3b.log; flags JJ_DONE (what day.sh waits for), E308_DONE. Starts runs only inside the GPU window.
cd "$(dirname "$0")/../.."
L=temp/e287/night3b.log
echo "waiting for temp/e287/LANES_DONE $(date)" >> $L
until [ -f temp/e287/LANES_DONE ]; do sleep 60; done
in_window() { H=$(date +%-H); [ "$H" -ge 22 ] || [ "$H" -lt 7 ] || { [ "$H" -ge 8 ] && [ "$H" -lt 17 ]; }; }
BEST=$(bash tools/env/_py14.sh temp/e287/pick_best.py 2>>$L | tail -1)
echo "E286 best arm: $BEST $(date)" >> $L
if ! in_window; then echo "outside the GPU window $(date), stopping" >> $L; exit 0; fi
export JJ_CTX=$(case $BEST in j0) echo none;; j1) echo dil;; j2) echo attn;; esac)
post308() {
  S=$1; R=e308_t_s$S
  [ -f runs/$R/model.pt ] || return
  bash tools/env/_pyr.sh jends_eval.py dump --run $R >> temp/e287/dump_$R.log 2>&1
  bash tools/env/_py14.sh jends_eval.py score --run $R --seed $S >> temp/e287/score_$R.log 2>&1
  bash tools/env/_py14.sh jpair_pr.py score --run $R --control e287_p2_s$S >> temp/e287/pr_$R.log 2>&1
  echo "E308 scored $S $(date)" >> $L
}
lane() {
  S=$1
  in_window || { echo "lane $S: window over $(date)" >> $L; return; }
  bash temp/e308/start_ntok.sh $S $BEST >> temp/e287/e308_lane$S.log 2>&1
  echo "E308 lane $S exit $? $(date)" >> $L
  post308 $S &
  in_window || { echo "lane $S: window over before jj $(date)" >> $L; wait; return; }
  bash temp/e286/start_jmaps.sh jj $S >> temp/e287/jj_lane$S.log 2>&1
  echo "jj lane $S exit $? $(date)" >> $L
  if [ -f runs/e286_jj_s$S/model.pt ]; then
    bash tools/env/_pyr.sh jmaps_eval.py score --run e286_jj_s$S >> temp/e287/jj_score_s$S.log 2>&1
    in_window || { echo "lane $S: window over before q2 $(date)" >> $L; wait; return; }
    bash temp/e287/start_pairs.sh q2 $S jj >> temp/e287/q2_lane$S.log 2>&1
    echo "q2 lane $S exit $? $(date)" >> $L
    if [ -f runs/e287_q2_s$S/model.pt ]; then
      bash tools/env/_pyr.sh jends_eval.py dump --run e287_q2_s$S >> temp/e287/dump_q2_s$S.log 2>&1
      bash tools/env/_py14.sh jends_eval.py score --run e287_q2_s$S --seed $S >> temp/e287/score_q2_s$S.log 2>&1
      echo "q2 scored $S exit $? $(date)" >> $L
    fi
  fi
  if in_window; then
    python3 ../autoresearch/tools/run.py --id E289.probe2_s$S --seed $S --gpu --retries 0 --no-card --split val -- bash tools/env/_pyr.sh shift_probe2.py $S >> temp/e289/probe2_s$S.log 2>&1
    echo "probe2 s$S exit $? $(date)" >> $L
  fi
  wait
}
lane 0 &
lane 1 &
wait
touch temp/e287/JJ_DONE
echo "done $(date)" >> $L
