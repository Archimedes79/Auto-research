#!/bin/bash
# The quiet-gap diagnostics chain (the lead's D119 / 23:30 orders + the report chat's E383 command list): measured numbers only, no card, no verdict, ONE slot, each job ALONE on the box (idle GPU, quiet host: the timing tools refuse otherwise).
# Starts only inside 04:30 - 07:00 or the weekday window 08:30 - 17:30 AND in a quiet gap (no other run.py --gpu job, no python wrapper of another job, no supervisor `sup*.sh`, load < 2.0, GPU idle <= 5 % / <= 2000 MiB, three checks a minute apart); a job that fails does not stop the chain (a twin killed by its timeout is expected).
# Jobs: (1) E379.time_e2 (the plain two-head ensemble: head alone / two in sequence / two stacked / whole path / parameters; registered by the E379 card) (2) S2 inference cost r vs z (`temp/e368/time_s2.py`) (3) S3 inference cost m vs z (`jmap_eval.py time`) (4) E383 list of the report chat: K 384 TRAIN cache, the speed-path GATE, the GPU tests, the early twins, the K 384 pair benchmark, the head timing at K 512 / K 384.
# NOT in this chain: the f arms of E383 (they need the gate, `gpu ok` twice, the twin curves, the lead's slot word: by hand); the S4 50-step gate (the S4 chain of Fable ran it). usage: bash temp/e379/quiet_chain.sh   (DRY=1: only say what it would run; FORCE=1: skip the window and gap checks, for the dry run)
cd "$(dirname "$0")/../.."
export BFLC_SIG=1.0
L=temp/e379/chain.log
say() { echo "$(date '+%m-%d %H:%M:%S') [quiet chain] $*" >> $L; }
exec 8>/tmp/lane_quiet_chain.lock
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
too_late() {  # the lead's rule (2026-11-18 02:50): NO new quiet-chain job after 06:45, the 07:00 landing has priority; what is left moves to the day window after the landing, on the landed tree
  local now=$((10#$(date +%H%M)))
  if [ -z "$FORCE" ] && [ $now -ge 645 ] && [ $now -lt 1200 ]; then say "$1: after 06:45: the chain stops here (the landing has priority); the rest runs in the day window after the landing"; exit 0; fi
}
job() {  # ID TIMEOUT_MIN CARDFLAG CMD...
  local id=$1 to=$2 card=$3; shift 3
  while [ -z "$FORCE" ] && ! gap; do too_late "$id"; sleep 30; done  # each job alone: wait for the gap again (the previous job's wrapper may still be alive for a second)
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
say "armed: waits for a quiet gap inside 04:30 - 07:00 or the weekday window 08:30 - 17:30"
n=0
until [ $n -ge 3 ]; do
  if [ -n "$FORCE" ] || { in_window && gap; }; then n=$((n + 1)); else n=0; fi
  [ $n -ge 3 ] && break
  sleep 60
done
say "quiet gap found: the chain starts"
HYP="E379 e2 cost gate (D106): the cost of the plain two-head ensemble at inference (head alone, two in sequence, two stacked, whole path, parameters)"
PRED="a second head in sequence adds about 5 ms to the 16.6 ms path (stored not-quiet timing); stacked less: the question is whether stacked is at or below E324 s's 19.7 ms"
RULE="by hand (D106, the lead decides): if the stacked two-head whole path is at or below 19.7 ms on a quiet host the ensemble e2 may be preferred over the distilled head d"
job E379.time_e2 20 --card-e2 bash tools/env/_pyr.sh token_time_ens.py --runs e373_cb_s0,e367_b25_s0 --b-run runs/e292_a_s0 --frames 24 --repeats 5
job diag.s2_timing 20 --no-card bash tools/env/_pyr.sh time_s2.py --runs e368_r_s0,e368_z_s0 --frames 24 --repeats 4
job diag.s3_time_m 20 --no-card bash tools/env/_pyr.sh jmap_eval.py time --run e374_m_s0
job diag.s3_time_z 20 --no-card bash tools/env/_pyr.sh jmap_eval.py time --run e374_z_s0
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
