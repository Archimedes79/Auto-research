#!/bin/bash
# E387 step 1 (CPU, before any f number exists; card section 1): (a) reproduce the stored z_s0 / r_s0 VAL numbers on the landed code, (b) the per-class TRAIN sweeps of the four STORED arms (e368 z_s0, z_s1, r_s0, r_s1) on their TRAIN every-3 dumps, the four own cells and the COMMON cell (z's recipe cell = choose-cell on the mean of the two z tables), each written to the run folders / temp/e387/cells.log BEFORE any dump of f exists.
# usage: bash temp/e387/step1.sh      log: temp/e387/chain.log
cd "$(dirname "$0")/../.."
export BFLC_SIG=1.0
source temp/e387/texts387.sh
L=temp/e387/chain.log
say() { echo "$(date '+%m-%d %H:%M:%S') [e387 step1] $*" >> $L; }
RUN="python3 ../autoresearch/tools/run.py"
PY="env OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 nice -n 10 bash tools/env/_py14.sh"
exec 8>/tmp/e387_step1.lock
flock -n 8 || { say "already runs"; exit 0; }
say "start"
for A in z_s0 r_s0; do
  S=${A#*_s}
  $RUN --id E387.repro_$A --seed $S --retries 0 --require-card --split val --timeout 60 --hyp "$HYP" --pred "$PRED" --rule "$RULE" -- bash -c "$PY jtoken_eval.py score --run e368_$A --seed $S --link-thr 0.1 --link-thr-long 0.1 --keep-thr 0.6 --no-none-gate"
  say "repro $A exit $?"
done
for A in z_s0 z_s1 r_s0 r_s1; do
  S=${A#*_s}
  $RUN --id E387.sweep_$A --seed $S --retries 0 --require-card --split val --timeout 120 --hyp "$HYP" --pred "$PRED" --rule "$RULE" -- bash -c "$PY jtoken_eval.py sweep --run e368_$A --seed $S --split train --every 3 --grid perclass --jobs 3"
  say "sweep $A exit $?"
done
for A in z_s0 z_s1 r_s0 r_s1; do
  CELL=$($PY jtoken_eval.py choose-cell --runs e368_$A | tail -1)
  say "own cell $A: $CELL (from its TRAIN table, before any f dump)"
  echo "$A $CELL" >> temp/e387/cells.log
done
mkdir -p temp/e387/common_cell
CELL=$($PY jtoken_eval.py choose-cell --runs e368_z_s0,e368_z_s1 --out-dir temp/e387/common_cell | tail -1)
say "COMMON cell (the mean of the two z TRAIN tables): $CELL (before any f dump)"
echo "common $CELL" >> temp/e387/cells.log
say "done"
