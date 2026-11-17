#!/bin/bash
# E378 read-out of one head. usage: bash temp/e378/read378.sh dump|cpu x_s<S>
#   dump: VAL + TRAIN-sample (every 3) dumps on the GPU (blocking, the supervisor's slot)
#   cpu : the per-class TRAIN sweep -> the pinned cell (choose378.py, logged BEFORE any VAL score) -> the VAL score at that cell against the champion at D115 (the verdict card)
#         -> the D115 cell against the champion (reported; skipped when the cell is D115) -> anatomy
cd "$(dirname "$0")/../.."
export BFLC_SIG=1.0
MODE=${1:?mode}; A=${2:?arm_s}; S=${A#*_s}; R=e378_$A
source temp/e378/texts378.sh
RUN="python3 ../autoresearch/tools/run.py"
say() { echo "$(date '+%m-%d %H:%M:%S') [read $A $MODE] $*" >> temp/e378/chain378.log; }
PY="env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nice -n 10 bash tools/env/_py14.sh"; PYR="bash tools/env/_pyr.sh"
if [ "$MODE" = dump ]; then
  $RUN --id E378.dump_$A --seed $S --gpu --retries 1 --require-card --split val --timeout 60 --hyp "$HYP" --pred "$PRED" --rule "$RULE" -- bash -c "$PYR jtoken_eval.py dump --run $R --seed $S --split val && $PYR jtoken_eval.py dump --run $R --seed $S --split train --every 3"
  say "exit $?"
elif [ "$MODE" = cpu ]; then
  $RUN --id E378.sweep_$A --seed $S --retries 0 --require-card --split val --timeout 120 --hyp "$HYP" --pred "$PRED" --rule "$RULE" -- bash -c "$PY jtoken_eval.py sweep --run $R --seed $S --split train --every 3 --grid perclass" || { say "sweep failed"; exit 1; }
  CELL=$(python3 temp/e378/choose378.py $R) || { say "chooser failed"; exit 1; }
  say "pinned cell $CELL (from the TRAIN table, before any VAL score)"
  IFS=, read LT LTL KT G <<< "$CELL"
  GATE=$([ "$G" = off ] && echo --no-none-gate)
  $RUN --id E378.read_$A --seed $S --retries 0 --require-card --split val --timeout 150 --hyp "$HYP" --pred "$PRED" --rule "$RULE" -- bash -c "$PY jtoken_eval.py score --run $R --seed $S --link-thr $LT --link-thr-long $LTL --keep-thr $KT $GATE --ref-run e373_cb_s$S --ref-run-cell 0.3,0.2,0.6,off && $PY jtoken_eval.py anatomy --run $R --seed $S --link-thr-long 0.2 --keep-thr 0.6"
  say "exit $?"
  if [ "$CELL" != "0.3,0.2,0.6,off" ]; then
    $RUN --id E378.d115_$A --seed $S --retries 0 --require-card --split val --timeout 60 --hyp "$HYP" --pred "$PRED" --rule "$RULE" -- bash -c "$PY jtoken_eval.py score --run $R --seed $S --link-thr 0.3 --link-thr-long 0.2 --keep-thr 0.6 --no-none-gate --ref-run e373_cb_s$S"
    say "d115 exit $?"
  fi
else
  say "unknown mode $MODE"; exit 1
fi
