#!/bin/bash
# The quiet-gap chain, 2nd version (2026-11-18 05:20): the timing jobs of chain 1 were flagged NOT QUIET only because `jtiming.gpu_state` was sampled at once after the timed passes (the after-sample read this process's own kernels, 65 - 70 %); `jtiming.py`'s own protocol waits 2.5 s first. The cost jobs run again WITH the wait (token_time_ens.py `after_state`, temp/e368/time_s2.py, temp/e374/time_s3_aged.py), then the E383 list of the report chat from `cache384` on, as chain 1 had it.
# Same rules as chain 1: starts only inside 04:30 - 07:00 or the weekday window 08:30 - 17:30 AND in a quiet gap (no other run.py --gpu job, no python wrapper of another job, no supervisor sup*.sh, load < 2.0, GPU idle <= 5 % / <= 2000 MiB, three checks a minute apart), no new job after 06:45 (the 07:00 landing has priority). usage: bash temp/e379/quiet_chain2.sh   (DRY=1, FORCE=1 as in chain 1)
cd "$(dirname "$0")/../.."
export BFLC_SIG=1.0
L=temp/e379/chain.log
say() { echo "$(date '+%m-%d %H:%M:%S') [quiet chain 2] $*" >> $L; }
exec 8>/tmp/lane_quiet_chain2.lock
flock -n 8 || { say "already runs"; exit 0; }
in_window() {
  local now=$((10#$(date +%H%M))) dow; dow=$(date +%u)
  { [ $now -ge 430 ] && [ $now -lt 700 ]; } || { [ "$dow" -le 5 ] && [ $now -ge 830 ] && [ $now -lt 1730 ]; }
}
gap() {
  pgrep -f "run.py .*--gpu" > /dev/null && return 1
  pgrep -f "tools/env/_py14.sh|tools/env/_pyr.sh" > /dev/null && return 1
  pgrep -f "(/|_)sup[A-Za-z0-9_]*\.sh" > /dev/null && return 1
  awk '{exit !($1 < 2.0)}' /proc/loadavg || return 1
  local g; g=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
  [ -n "$g" ] || return 1
  local u=${g%%,*} m=${g##*, }
  [ "${u// /}" -le 5 ] && [ "${m// /}" -le 2000 ]
}
RUN="python3 ../autoresearch/tools/run.py"
too_late() {
  local now=$((10#$(date +%H%M)))
  if [ -z "$FORCE" ] && [ $now -ge 645 ] && [ $now -lt 1200 ]; then say "$1: after 06:45: the chain stops here (the landing has priority); the rest runs in the day window after the landing"; exit 0; fi
}
job() {  # ID TIMEOUT_MIN CARDFLAG CMD...
  local id=$1 to=$2 card=$3; shift 3
  while [ -z "$FORCE" ] && ! gap; do too_late "$id"; sleep 30; done
  too_late "$id"
  say "start $id"
  if [ -n "$DRY" ]; then say "DRY: $*"; return 0; fi
  if [ "$card" = "--card-e2" ]; then
    $RUN --id "$id" --seed 0 --gpu --ram-mb 6000 --retries 0 --require-card --hyp "$HYP" --pred "$PRED" --rule "$RULE" --split val --timeout "$to" -- "$@" >> temp/e379/quiet_chain.out 2>&1
  else
    $RUN --id "$id" --seed 0 --gpu --ram-mb 6000 --retries 0 $card --split val --timeout "$to" -- "$@" >> temp/e379/quiet_chain.out 2>&1
  fi
  say "$id exit $?"
}
say "armed (the cost jobs again with the aging wait, then the E383 list): waits for a quiet gap inside 04:30 - 07:00 or the weekday window 08:30 - 17:30"
n=0
until [ $n -ge 3 ]; do
  if [ -n "$FORCE" ] || { in_window && gap; }; then n=$((n + 1)); else n=0; fi
  [ $n -ge 3 ] && break
  sleep 60
done
say "quiet gap found: the chain starts"
HYP="E379 e2 cost gate (D106), re-run with the aging wait before the after-sample (the first run was flagged NOT QUIET by the tool's own sampling artefact): the cost of the plain two-head ensemble at inference (head alone, two in sequence, two stacked, whole path, parameters)"
PRED="a second head in sequence adds about 5 ms to the 16.6 ms path (stored timing); stacked less: the question is whether stacked is at or below E324 s's 19.7 ms"
RULE="by hand (D106, the lead decides): if the stacked two-head whole path is at or below 19.7 ms on a quiet host the ensemble e2 may be preferred over the distilled head d"
job E379.time_e2b 20 --card-e2 bash tools/env/_pyr.sh token_time_ens.py --runs e373_cb_s0,e367_b25_s0 --b-run runs/e292_a_s0 --frames 24 --repeats 5
job diag.s2_timing_b 20 --no-card bash tools/env/_pyr.sh time_s2.py --runs e368_r_s0,e368_z_s0 --frames 24 --repeats 4
job diag.s3_time_m_b 20 --no-card bash tools/env/_pyr.sh time_s3_aged.py --run e374_m_s0
job diag.s3_time_z_b 20 --no-card bash tools/env/_pyr.sh time_s3_aged.py --run e374_z_s0
job diag383.cache384 30 --no-card bash tools/env/_pyr.sh build_tokcache_root.py --root runs/e383_cache --run runs/e292_a_s0 --split train --K 384 --no-f2
job diag383.bench_gate 20 --no-card bash tools/env/_py14.sh bench_token_step.py gate
job diag383.gputests 10 --no-card bash -c "JTF_GPU=1 bash tools/env/_py14.sh test_jtoken_fast.py gpu && JTF_GPU=1 bash tools/env/_py14.sh test_lane_ptok_gpu.py gpu"
job diag383.twin_plain 8 --no-card bash tools/env/_pyr.sh jtoken_train.py --name e383_twin_plain --seed 0 --head-seed 2 --ident e292_a --steps 12000 --bs 8 --lr 1e-3 --workers 4 --none-weight 0.25
job diag383.twin_fast 8 --no-card bash tools/env/_pyr.sh jtoken_fast_train.py --name e383_twin_fast --seed 0 --head-seed 2 --ident e292_a --steps 12000 --bs 8 --lr 1e-3 --workers 4 --none-weight 0.25
if [ -z "$DRY" ]; then
  python3 tools/jarm.py curve e383_twin_fast e383_twin_plain --band 100:0.075,300:0.04,500:0.025,600:0.025 >> temp/e379/quiet_chain.out 2>&1; say "curve fast vs plain exit $?"
  python3 tools/jarm.py curve e383_twin_plain e373_cb_s0 --band 100:0.075,300:0.04,500:0.025,600:0.025 >> temp/e379/quiet_chain.out 2>&1; say "curve plain vs e373 exit $?"
fi
job diag383.bench_pair 20 --no-card bash tools/env/_py14.sh bench_token_step.py pair
job diag383.timing512 20 --no-card bash tools/env/_py14.sh jtoken_eval.py timing --run e373_cb_s0 --b-run runs/e292_a_s0 --K 512
job diag383.timing384 20 --no-card bash tools/env/_py14.sh jtoken_eval.py timing --run e373_cb_s0 --b-run runs/e292_a_s0 --K 384
say "chain done (the f arms of E383 are NOT part of it)"
