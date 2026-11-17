#!/bin/bash
# E377 read-out of one head (card v2.1). usage: bash temp/e377/read377.sh dump|cpu n_s<S>
#   dump: the VAL dump and the TRAIN-sample dump (every 3) on the GPU through run.py (blocking, on the supervisor's slot)
#   cpu : (a) the D115 cell against the champion head (cross-run interval), (c) argmax at keep 0.6, the anatomy (incl. the near / far table), the per-class TRAIN sweep for the pinned (b) rule
cd "$(dirname "$0")/../.."
export BFLC_SIG=1.0
MODE=${1:?mode}; A=${2:?arm_s}; S=${A#*_s}; R=e377_$A
source temp/e377/texts377.sh
RUN="python3 ../autoresearch/tools/run.py"
say() { echo "$(date '+%m-%d %H:%M:%S') [read $A $MODE] $*" >> temp/e377/chain377.log; }
PY="env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nice -n 10 bash tools/env/_py14.sh"; PYR="bash tools/env/_pyr.sh"
D115="--link-thr 0.3 --link-thr-long 0.2 --keep-thr 0.6 --no-none-gate"
if [ "$MODE" = dump ]; then
  $RUN --id E377.dump_$A --seed $S --gpu --retries 1 --require-card --split val --timeout 60 --hyp "$HYP" --pred "$PRED" --rule "$RULE" -- bash -c "$PYR jtoken_eval.py dump --run $R --seed $S --split val && $PYR jtoken_eval.py dump --run $R --seed $S --split train --every 3"
  say "exit $?"
elif [ "$MODE" = cpu ]; then
  $RUN --id E377.read_$A --seed $S --retries 0 --require-card --split val --timeout 150 --hyp "$HYP" --pred "$PRED" --rule "$RULE" -- bash -c "$PY jtoken_eval.py score --run $R --seed $S $D115 --ref-run e373_cb_s$S && $PY jtoken_eval.py score --run $R --seed $S --argmax --keep-thr 0.6 && $PY jtoken_eval.py anatomy --run $R --seed $S --link-thr-long 0.2 --keep-thr 0.6 && $PY jtoken_eval.py sweep --run $R --seed $S --split train --every 3 --grid perclass"
  say "exit $?"
else
  say "unknown mode $MODE"; exit 1
fi
