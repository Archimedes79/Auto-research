#!/bin/bash
# E379 gate job: the cost of the plain two-head ensemble (E379.time_e2, `evaluation/token_time_ens.py`) on a QUIET host and an idle GPU. The lead's order (2026-11-17 20:40): it is the LAST GPU job of the night: it starts in the first gap after the S4 chain and the E379 arms,
# pausing / killing nothing; if the night gives no gap before 07:00 it runs at the start of the next window with nothing else started. Windows: the GPU night 22:00 - 07:00 (earliest start here 04:30) and the weekday day window 08:30 - 17:30.
# A gap = no other `run.py --gpu` job, no python wrapper (`_py14.sh` / `_pyr.sh`) of another job, no night supervisor (`sup*.sh`), load average (1 min) below 2.0, GPU idle (util <= 5 %, memory <= 2000 MiB), on THREE checks one minute apart (a supervisor's 20 s
# gap between two jobs is no gap). The job goes through run.py (card E379, arm e2's run id); a result with usable_for_gcost false (not quiet at the end) is repeated up to 3 times. usage: bash temp/e379/time_e2_waiter.sh   (DRY=1: say what it would do)
cd "$(dirname "$0")/../.."
export BFLC_SIG=1.0
source tools/maintenance/keep_alive.sh
ka_init temp/e379/chain.log e379_time_e2 lane_e379_time_e2_waiter
HYP="E379 e2 cost gate (D106): what does the plain ensemble of the two champion-recipe heads cost at inference: the head alone, two heads in sequence, two heads stacked in one forward (vmap), each alone and behind B' (the whole path), parameters"
PRED="head about 5 ms of the whole path (stored not-quiet timing: head 6.6 ms, B' 11.4, whole 16.6); two heads in sequence about 21.8 ms; stacked below that (launch-bound head): the question is whether stacked is at or below E324 s's 19.7 ms"
RULE="by hand (D106, the lead decides): if the STACKED two-head whole path is at or below 19.7 ms on a quiet host (usable_for_gcost true) the plain ensemble e2 may be preferred as the simpler champion over the distilled head d; otherwise d keeps the gain at one head's cost"
in_window() { ka_in_window 0430 0700 || ka_in_window 0830 1730; }
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
say "armed: waits for a quiet gap inside 04:30 - 07:00 or the weekday day window 08:30 - 17:30"
n=0
while [ $n -lt 3 ]; do
  ok=0
  until [ $ok -ge 3 ]; do
    if in_window && gap; then ok=$((ok + 1)); else ok=0; fi
    [ $ok -ge 3 ] && break
    sleep 60
  done
  n=$((n + 1))
  say "quiet gap found, attempt $n"
  if [ -n "$DRY" ]; then say "DRY: would run E379.time_e2"; exit 0; fi
  python3 ../autoresearch/tools/run.py --id E379.time_e2 --seed 0 --gpu --ram-mb 4000 --retries 0 --require-card --split val --timeout 20 --hyp "$HYP" --pred "$PRED" --rule "$RULE" -- bash tools/env/_pyr.sh token_time_ens.py --runs e373_cb_s0,e367_b25_s0 --b-run runs/e292_a_s0 --frames 24 --repeats 5 >> temp/e379/time_e2.out 2>&1
  say "attempt $n: run.py exit $?"
  f=$(ls -t runs/e373_cb_s0/token_timing_ens_2*.json 2>/dev/null | head -1)
  if [ -n "$f" ] && grep -q '"usable_for_gcost": true' "$f"; then say "usable result: $f"; exit 0; fi
  say "no usable (quiet) result yet ($f); waiting for the next gap"
  sleep 300
done
say "3 attempts without a usable result: giving up (the numbers of the last file are not quiet-host numbers)"
exit 1
