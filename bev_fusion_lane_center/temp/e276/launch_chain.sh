#!/bin/bash
# E276: the GPU job scheduler, inside Mirko's windows only (nights 22:00-07:00, Monday-Friday 08:30-17:30). It starts when temp/e276/READY exists (made after the full l4 store is checked), and starts a job only
# when (1) the window has enough minutes left for it, (2) fewer than 2 lane GPU jobs (run.py launchers of E275 / E276) run, (3) less than 8.5 GB of GPU memory is used, (4) (a BEV job would show in the GPU memory)
# (D57 / D61: this chat has priority, the BEV chat holds its chain while READY exists), (5) the jobs it depends on are done. Jobs in priority order (arm:seed:minutes of window needed:dependencies):
#   mt / ms = a GPU smoke of arm t / arm s (20 steps per phase, no card, never a verdict), t / s = training of the arm (phases 3-7 from the E273 phase-2 checkpoint), c = score B on GT l4,
#   tq / sq = score the arm (card + full dump for the video-note analyses). A failed job stops the scheduling (running jobs finish); verdicts run by themselves when their jobs are done.
# Log temp/e276/chain.log; flags temp/e276/STARTED_<arm><seed>, VERDICT_T_DONE, VERDICT_S_DONE, CHAIN_DONE (only when every job is done).
cd "$(dirname "$0")/../.."
L=temp/e276/chain.log
JOBS="mt:0:35: ms:0:35: t:0:260:mt0,ms0 t:1:260:mt0,ms0 c:0:55: c:1:55: tq:0:55:t0 tq:1:55:t1 u:0:120: u:1:120: qp:0:40: qp:1:40: uq:0:55:u0 uq:1:55:u1 w:0:120: w:1:120: wq:0:55:w0 wq:1:55:w1"   # 2026-11-08 01:10 (u / uq = E277, the soft close-node step, added 01:40): the s arms are out (s_s0 failed its phase-3 gate at 00:50: delivered 3.6 nodes per 100 m at e 0.5, recall 0.15); qp = E278 diagnostic (phase-6 models scored); the E277 jobs are added when the code is ready
now_min() { echo $((10#$(date +%H) * 60 + 10#$(date +%M))); }
window_left() {                                   # minutes left in the current window (0 = outside)
  d=$(date +%u); n=$(now_min)
  if [ $n -ge 1320 ]; then echo $((1440 - n + 420)); return; fi
  if [ $n -lt 420 ]; then echo $((420 - n)); return; fi
  if [ $d -le 5 ] && [ $n -ge 510 ] && [ $n -lt 1050 ]; then echo $((1050 - n)); return; fi
  echo 0
}
mem_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1; }
lane_jobs() { pgrep -fc "tools/run.py --id E27[5-9]\." ; }

rdir() { ls -d experiment_execution/runs/E27[5-9].$1 2>/dev/null | head -1; }                 # E276.t_s0, E277.u_s0, E278.qp_s0, ...: the arm names are unique over the night
done_of() { d=$(rdir $1); [ -n "$d" ] && [ -f "$d/done" ]; }
id_of() { echo "${1%%[0-9]*}_s${1##*[a-z]}"; }              # t0 -> t_s0
failed_of() { d=$(rdir $1); [ -n "$d" ] && grep -qE '"status": "(failed|stalled|stopped|blocked|held)"' $d/run.json 2>/dev/null; }
deps_done() {
  [ -z "$1" ] && return 0
  for d in ${1//,/ }; do done_of "$(id_of $d)" || return 1; done
  return 0
}
echo "scheduler armed $(date)" > $L
while true; do
  [ -f temp/e276/READY ] || { sleep 30; continue; }
  # the E275 launcher gates ONCE on an idle GPU before its first run (no trainer, < 2.6 GB): let it start first (D57: its scoring is short), then both run side by side
  if pgrep -f launch_scoring > /dev/null && ! grep -q '^start' temp/e275/launch.log 2>/dev/null && [ ! -f temp/e275/SCORED ] && [ "$(window_left)" -gt 0 ]; then
    [ -z "$E275_WAIT_SINCE" ] && E275_WAIT_SINCE=$(date +%s)
    # fallback: if the E275 launcher has not started 30 minutes into the window (something else holds the GPU), go on without it (the 22:43 check of the lane chat looks at E275 then)
    if [ $(( $(date +%s) - E275_WAIT_SINCE )) -lt 1800 ]; then sleep 20; continue; fi
  fi
  all_done=1; failed=0
  for job in $JOBS; do
    IFS=: read arm seed need deps <<< "$job"
    id=$(id_of "$arm$seed")
    done_of "$id" && continue
    all_done=0
    if [ -f temp/e276/STARTED_$arm$seed ] && failed_of "$id"; then failed=1; echo "$(date): $id failed: no further job is started" >> $L; fi
  done
  [ $failed = 1 ] && exit 1
  if [ ! -f temp/e276/VERDICT_T_DONE ] && done_of t_s0 && done_of t_s1 && done_of tq_s0 && done_of tq_s1; then
    { echo "== verdict276 t against B $(date)"; OMP_NUM_THREADS=2 bash tools/env/_py14.sh qgraph_run_eval.py verdict276 --kind t --var e276_t_s0 e276_t_s1 --ctl e273_p_s0 e273_p_s1; } >> temp/e276/verdicts.log 2>&1
    touch temp/e276/VERDICT_T_DONE; echo "$(date): verdict t written" >> $L
  fi
  if [ ! -f temp/e276/VERDICT_S_DONE ] && done_of s_s0 && done_of s_s1 && done_of sq_s0 && done_of sq_s1 && done_of t_s0 && done_of tq_s0; then
    { echo "== verdict276 s against t $(date)"; OMP_NUM_THREADS=2 bash tools/env/_py14.sh qgraph_run_eval.py verdict276 --kind s --var e276_s_s0 e276_s_s1 --ctl e276_t_s0 e276_t_s1; } >> temp/e276/verdicts.log 2>&1
    touch temp/e276/VERDICT_S_DONE; echo "$(date): verdict s written" >> $L
  fi
  if [ ! -f temp/e276/VERDICT_U_DONE ] && done_of u_s0 && done_of u_s1 && done_of uq_s0 && done_of uq_s1; then
    { echo "== verdict277 u against B and against B + hard step $(date)"; OMP_NUM_THREADS=2 bash tools/env/_py14.sh qgraph_run_eval.py verdict277 --var e277_u_s0.plain_t0.5 e277_u_s1.plain_t0.5 --ctl e273_p_s0 e273_p_s1 --hard e273_p_s0.close1.5_t0.5 e273_p_s1.close1.5_t0.5; } >> temp/e276/verdicts.log 2>&1
    touch temp/e276/VERDICT_U_DONE; echo "$(date): verdict277 written" >> $L
  fi
  if [ ! -f temp/e276/VERDICT_W_DONE ] && done_of w_s0 && done_of w_s1 && done_of wq_s0 && done_of wq_s1; then
    { echo "== verdict277 for E279 (radius 1.25): w against B and against B + hard step $(date)"; OMP_NUM_THREADS=2 bash tools/env/_py14.sh qgraph_run_eval.py verdict277 --var e279_w_s0.plain_t0.5 e279_w_s1.plain_t0.5 --ctl e273_p_s0 e273_p_s1 --hard e273_p_s0.close1.5_t0.5 e273_p_s1.close1.5_t0.5 --exp-var e279; } >> temp/e276/verdicts.log 2>&1
    touch temp/e276/VERDICT_W_DONE; echo "$(date): verdict277 (E279) written" >> $L
  fi
  if [ $all_done = 1 ]; then break; fi                                # after the verdict blocks (2026-11-08: the first version broke out before the E277 verdict was written)
  if [ "$(lane_jobs)" -lt 2 ] && [ "$(mem_used)" -lt 8500 ]; then
    for job in $JOBS; do
      IFS=: read arm seed need deps <<< "$job"
      [ -f temp/e276/STARTED_$arm$seed ] && continue
      deps_done "$deps" || continue
      [ "$(window_left)" -ge "$need" ] || continue
      touch temp/e276/STARTED_$arm$seed
      echo "$(date): start $arm seed $seed (window left $(window_left) min, GPU $(mem_used) MB, lane jobs $(lane_jobs))" >> $L
      case $arm in qp) script=start_arm2.sh;; u|uq) script=start_arm3.sh;; w|wq) script=start_arm4.sh;; *) script=start_arm.sh;; esac
      nohup bash temp/e276/$script $arm $seed > temp/e276/start_${arm}_s${seed}.log 2>&1 &
      sleep 60                                    # run.py claims its slot
      break
    done
  fi
  sleep 30
done
touch temp/e276/CHAIN_DONE
echo "$(date): every job is done" >> $L
