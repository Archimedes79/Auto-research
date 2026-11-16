#!/bin/bash
# Day chain 33 (lane experiments chat; card E352 STEP 0 on B): the point set of day30 in an ORDER BY VALUE, worked by DAY33_WORKERS parallel streams (default 3), CPU only, every job a run.py job (hard rule 3).
# Why (Mirko's order of 2026-11-15 22:00: every experiment finished by Monday 07:30): the serial day30 chain would reach about 12 of its 22 stages by then; this order puts the stages that decide the rule first:
#   null s0, null s1 (the gate on both seeds, 3 read-outs) -> T1, T2 on seed 0 (whole GroupNorm groups) -> the random-group control on seed 0 (r1d0-2, r2d0-2: the kill rule needs the mean of 3 draws) -> T1, T2 on seed 1 -> random seed 1 -> the channel-level anchor.
# The functions, the point set and the markers are those of day30.sh (sourced: temp/day30/*.done are shared, a stage finished there stays finished). A stage is claimed by a worker with a directory temp/day30/claim_<stage> (pid inside; a dead pid is taken over).
# The table is NOT a stage: run `bash tools/env/_py14.sh jprune_b.py table --seeds 0 [1] --out temp/e352/step0_table.md` by hand at any time (it tolerates missing points).
# Start: waits for DAY33_START_EPOCH (default tomorrow 00:20) AND for the end of the E354 trainings (no `python jrecur_train.py`), then resumes the SIGSTOPped workers of day30 once (temp/day30/resumed.flag). DAY33_NOW=1 skips the wait. DAY30_DRY=1 prints the plan.
# The old day30.sh main shell must be dead (its run.py of the in-flight null card may live on: the worker below waits for a running id and takes the finished record).
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source temp/day30/day30.sh                     # functions only (sourced: nothing is started); sets D, THREADS, TIMEOUT, HYP, PRED, RULE
L=$D/day33.log
mkdir -p $D
WORKERS=${DAY33_WORKERS:-3}

# a job whose run id is running right now (record status running with a fresh heartbeat) is waited for, not started twice
running_id() {
  python3 - "$1" <<'EOF'
import json, os, sys, time
d = "experiment_execution/runs/" + sys.argv[1]
try:
    st = json.load(open(d + "/run.json")).get("status")
    age = time.time() - os.path.getmtime(d + "/heartbeat")
except Exception:
    sys.exit(1)
sys.exit(0 if st in ("running", "waiting") and age < 300 else 1)
EOF
}
eval "orig_$(declare -f job)"
job() { local id=$1; while running_id $id; do sleep 30; done; orig_job "$@"; }

# the ordered stage list: name, function, arguments (as day30.sh calls them)
STAGES=(
  "null_s0 null_gate 0"
  "null_s1 null_gate 1"
  "g_t1_s0 point 0 t1 1"
  "g_t2_s0 point 0 t2 1"
  "g_r1d0_s0 point 0 r1d0 1" "g_r1d1_s0 point 0 r1d1 1" "g_r1d2_s0 point 0 r1d2 1"
  "g_r2d0_s0 point 0 r2d0 1" "g_r2d1_s0 point 0 r2d1 1" "g_r2d2_s0 point 0 r2d2 1"
  "g_t1_s1 point 1 t1 1"
  "g_t2_s1 point 1 t2 1"
  "g_r1d0_s1 point 1 r1d0 1" "g_r1d1_s1 point 1 r1d1 1" "g_r1d2_s1 point 1 r1d2 1"
  "g_r2d0_s1 point 1 r2d0 1" "g_r2d1_s1 point 1 r2d1 1" "g_r2d2_s1 point 1 r2d2 1"
  "c_t1_s0 point 0 t1 0" "c_t2_s0 point 0 t2 0" "c_t1_s1 point 1 t1 0" "c_t2_s1 point 1 t2 0"
)

claim() { # claim <stage>: 0 when this worker holds the stage now
  local c=$D/claim_$1
  if mkdir $c 2>/dev/null; then echo $$ > $c/pid; return 0; fi
  local p; p=$(cat $c/pid 2>/dev/null)
  if [ -n "$p" ] && ! kill -0 "$p" 2>/dev/null; then rm -rf $c; mkdir $c 2>/dev/null && { echo $$ > $c/pid; return 0; }; fi
  return 1
}

worker() {
  local W=$1 spec name
  for spec in "${STAGES[@]}"; do
    name=${spec%% *}
    [ -f $D/$name.done ] && continue
    if ls $D/FAIL_* >/dev/null 2>&1; then say "w$W: a stage failed (FAIL_* marker), this worker stops"; return 1; fi
    claim $name || continue
    say "w$W: claims $name"
    stage $spec                                # stage exits the shell on failure: this worker ends, the other workers see the FAIL_ marker
    rm -rf $D/claim_$name
  done
  say "w$W: nothing left"
}

if [ -n "$DAY30_DRY" ]; then
  for spec in "${STAGES[@]}"; do say "DRY stage $spec"; done
  point 0 t1 1                                  # one point of the list through the real functions (one python start): the run.py lines it would start
  exit 0
fi
if [ "${BASH_SOURCE[0]}" != "$0" ]; then return 0 2>/dev/null; fi
exec 8>/tmp/day33.lock
flock -n 8 || { echo "day33 already runs $(date)" >> $L; exit 0; }
START=${DAY33_START_EPOCH:-$(date -d "tomorrow 00:20" +%s)}
say "day33 starts: $WORKERS workers, wait until $(date -d @$START +%H:%M) and for the end of the E354 trainings (DAY33_NOW=${DAY33_NOW:-no})"
if [ -z "$DAY33_NOW" ]; then until [ $(date +%s) -ge $START ] && ! pgrep -f "^python jrecur_train.py" >/dev/null; do sleep 30; done; fi
if [ ! -f $D/resumed.flag ]; then
  for p in $(pgrep -f "^python jprune_b.py"; pgrep -f "^python jends_eval.py score --run e352"); do kill -CONT "$p" 2>/dev/null && say "SIGCONT $p"; done
  touch $D/resumed.flag
fi
say "day33 go"
for W in $(seq 1 $WORKERS); do
  ( worker $W ) &
  sleep 20                                      # the claims and the first run.py starts do not collide
done
wait
say "day33 done"
