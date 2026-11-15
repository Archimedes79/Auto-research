#!/bin/bash
# Day chain 30 (lane experiments chat; card E352, STEP 0 on B): ONE serial chain, CPU only (nice -n 19), every job a run.py job (hard rule 3). Probe: evaluation/jprune_b.py (dump, gate, table), scoring: jends_eval.py score --full <B' dump> --base <B ckpt> --tag.
#   order: the NULL of seed 0 (dump + the 3 cards snap / primary / attach + the gate), the NULL of seed 1 (same), then the JUDGED set, whole GroupNorm groups: t1, t2 per seed (dump + 3 cards each), then the random-group draws r1d0-2 (T1) and r2d0-2 (T2) per seed, then the
#   channel-level E345 anchor t1, t2 per seed (the anchor's null is the null above: the same pass), then the table. Every job after the null jobs is `--after` the null gate of its seed (a failed gate stops the chain: the stage fails, nothing after it runs).
# Stages and jobs are idempotent: temp/day30/<stage>.done / <run id>.done are skipped; a failed stage stops the chain (log temp/day30/day30.log, FAIL_<stage>). Dry run: DAY30_DRY=1 bash temp/day30/day30.sh (prints the run.py lines, runs nothing). Sourcing the file only defines the functions.
# SIGSTOP / SIGCONT of the python processes (the E354 gate): harmless, nothing is polled or timed inside them but run.py's wall-clock --timeout (DAY30_TIMEOUT minutes, default 120 per job; a dump takes 12 - 25 minutes); the chain itself is bash and keeps waiting.
# Threads of the dump: DAY30_THREADS (default 1: see the measurement in temp/e353/build_notes.md). The box is loaded by two trainings: nice 19, no timing claim. Never edit this script while it runs.
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
D=temp/day30
[ -n "$DAY30_DRY" ] && D=temp/day30/dry_$(date +%H%M%S)
L=$D/day30.log
R=$PWD
THREADS=${DAY30_THREADS:-1}
TIMEOUT=${DAY30_TIMEOUT:-120}
HYP="E352 step 0 (card experiments/E352_pyramid-by-importance, section 0), no training: replacing the lowest-importance WHOLE GroupNorm groups of B's pyramid trunk (T1: e2 50 %, d2 25 %, e3 25 %; T2: T1 + d0, e1, d1 at 25 %) by their calibration means (the E345 hook edit, applied to B's own QGraph AND to the junction system) costs B's plain graph and the junction system on B' little, and less than replacing the same number of random whole groups of the same units; the null (no edit) reproduces the stored cards of E324 s."
PRED="Against the null of the same seed, topo F1 of B's plain graph: T1 -0.004 (window -0.012 to +0.001), T2 -0.010 (window -0.025 to -0.002); the mean of the 3 random-group draws is worse than the ranked set by at least 0.003 on B's plain graph, per seed and set (window 0 to 0.02); junction snapped (B' as the delivered graph) T1 -0.004 (window -0.010 to +0.002), T2 -0.009 (window -0.018 to -0.002; this T2 window is the coder's proposal); the null reproduces the stored cards of E324 s on snap / primary / attach with topo F1 within 2e-4 and every other number within max(2e-3, 2e-3 x stored)."
RULE="E352 card section 0 (by hand: jprune_b.py table over the cards, no tool decides): KILL when T1 loses more than 0.03 topo F1 on B's plain graph against the null and the mean of the random draws loses no more (judged per seed on the groups variant); the NULL GATE (jprune_b.py gate: topo F1 2e-4, every other number max(2e-3, 2e-3 x stored), both seeds, three read-outs) must hold before any other point counts; the channel-level points (e352c) are the E345 anchor, context only; VAL 873 frames, no TEST."

say() { echo "$* $(date)" >> $L; [ -n "$DAY30_DRY" ] && echo "$*"; }
# okrun <run id>: the run record says the job finished with exit 0 (a chain killed between the end of a job and its .done marker must not start the id again: run.py refuses a finished id)
okrun() { python3 -c "
import json,sys
try: st=json.load(open('experiment_execution/runs/$1/run.json')).get('status')
except Exception: st=None
sys.exit(0 if st=='ok' else 1)"; }
# job <run id> <seed> <after ids or -> <command ...>: one run.py job (blocks until it ends); a job with a .done marker (or an 'ok' run record) is skipped. --timeout counts the wall time INCLUDING a pause: a SIGSTOP of the dump of up to about 100 minutes is harmless, a longer one times
# the job out (exit 124, the stage fails, the chain stops); starting day30.sh again resumes at that job (run.py resumes a failed id, the dump begins again: it writes only at its end)
job() {
  local id=$1 seed=$2 after=$3; shift 3
  [ -f $D/$id.done ] && return 0
  local af=""; [ "$after" != "-" ] && af="--after $after"
  if [ -n "$DAY30_DRY" ]; then say "DRY run.py --id $id --seed $seed $af --timeout $TIMEOUT -- nice -n 19 $*"; return 0; fi
  okrun $id && { touch $D/$id.done; return 0; }
  python3 ../autoresearch/tools/run.py --id $id --seed $seed --retries 0 --require-card --split val --timeout $TIMEOUT $af --hyp "$HYP" --pred "$PRED" --rule "$RULE" -- nice -n 19 "$@" && touch $D/$id.done
}
stage() { local n=$1; shift; [ -f $D/$n.done ] && return 0; say "stage $n starts"; if "$@"; then touch $D/$n.done; say "stage $n done"; else say "stage $n FAILED"; touch $D/FAIL_$n; exit 1; fi; }
vl() { case "$1:$2" in null:*) echo "";; *:1) echo g;; *) echo c;; esac; }          # the variant letter of a point: none for the null, g whole groups, c channels (as jprune_b.run_id)
# point <seed> <point> <groups 0|1>: the dump and the three cards of one point, serial
point() {
  local S=$1 P=$2 G=$3 v id gf="" after="-"
  v=$(vl $P $G); [ "$G" = 1 ] && gf="--groups"
  id=E352.s0${v}_${P}
  [ "$P" != null ] && after=E352.s0_null_gate_s$S
  job ${id}_s$S $S $after bash tools/env/_py14.sh jprune_b.py dump --seed $S --point $P $gf --threads $THREADS || return 1
  local lines line
  lines=$(bash tools/env/_py14.sh jprune_b.py commands --seed $S --point $P $gf 2>/dev/null)         # the scoring lines of the point (one python start)
  for K in snap primary attach; do
    line=$(echo "$lines" | grep -A1 "^# $P $K\$" | tail -1)
    case "$line" in "bash tools/env/_py14.sh jends_eval.py score "*) ;; *) say "no scoring line for $P $K"; return 1;; esac
    job ${id}_${K}_s$S $S ${id}_s$S $(eval echo "$line") || return 1
  done
}
null_gate() { # the null of a seed: dump, cards, the gate (a failed gate fails the stage and stops the chain)
  local S=$1
  point $S null 0 || return 1
  job E352.s0_null_gate_s$S $S E352.s0_null_snap_s$S,E352.s0_null_primary_s$S,E352.s0_null_attach_s$S bash tools/env/_py14.sh jprune_b.py gate --seed $S
}

if [ "${BASH_SOURCE[0]}" != "$0" ]; then return 0 2>/dev/null; fi        # sourced: functions only, nothing is started
mkdir -p $D
exec 9>/tmp/day30${DAY30_DRY:+_dry}.lock
flock -n 9 || { echo "day30 already runs $(date)" >> $L; exit 0; }
say "day30 starts (dry: ${DAY30_DRY:-no}, dump threads $THREADS)"
stage null_s0 null_gate 0
stage null_s1 null_gate 1
for S in 0 1; do for P in t1 t2; do stage g_${P}_s$S point $S $P 1; done; done
for S in 0 1; do for P in r1d0 r1d1 r1d2 r2d0 r2d1 r2d2; do stage g_${P}_s$S point $S $P 1; done; done
for S in 0 1; do for P in t1 t2; do stage c_${P}_s$S point $S $P 0; done; done
stage table job E352.s0_table 0 - bash tools/env/_py14.sh jprune_b.py table --seeds 0 1 --out temp/e352/step0_table.md
say "day30 done"
touch $D/DAY30_DONE
