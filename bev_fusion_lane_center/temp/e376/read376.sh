#!/bin/bash
# E376 read-out of one head (card section 0). usage: bash temp/e376/read376.sh dump|cpu <arm>_s<S>
#   dump: the VAL dump and the TRAIN-sample dump (every 3 = 291 frames) on the GPU, through run.py (the supervisor runs it on its own slot, blocking)
#   cpu : (a) the VAL score at the D115 per-class decode (short 0.3 / long 0.2 / keep 0.6 / none gate off) and the per-class TRAIN sweep (36 cells) for read-out (b)
cd "$(dirname "$0")/../.."
export BFLC_SIG=1.0
MODE=${1:?mode}; A=${2:?arm_s}; S=${A#*_s}; R=e376_$A
source temp/e376/texts376.sh
RUN="python3 ../autoresearch/tools/run.py"
say() { echo "$(date '+%m-%d %H:%M:%S') [read $A $MODE] $*" >> temp/e376/chain376.log; }
PY="env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nice -n 10 bash tools/env/_py14.sh"; PYR="bash tools/env/_pyr.sh"
D115="--link-thr 0.3 --link-thr-long 0.2 --keep-thr 0.6 --no-none-gate"
if [ "$MODE" = dump ]; then
  $RUN --id E376.dump_$A --seed $S --gpu --retries 1 --require-card --split val --timeout 60 --hyp "$HYP" --pred "$PRED" --rule "$RULE" -- bash -c "$PYR jtoken_eval.py dump --run $R --seed $S --split val && $PYR jtoken_eval.py dump --run $R --seed $S --split train --every 3"
  say "exit $?"
elif [ "$MODE" = cpu ]; then
  $RUN --id E376.read_$A --seed $S --retries 0 --require-card --split val --timeout 150 --hyp "$HYP" --pred "$PRED" --rule "$RULE" -- bash -c "$PY jtoken_eval.py score --run $R --seed $S $D115 --ref-run e373_cb_s$S && $PY jtoken_eval.py score --run $R --seed $S --argmax --keep-thr 0.6 && $PY jtoken_eval.py anatomy --run $R --seed $S --link-thr-long 0.2 --keep-thr 0.6 && $PY jtoken_eval.py sweep --run $R --seed $S --split train --every 3 --grid perclass"
  say "exit $?"
else
  say "unknown mode $MODE"; exit 1
fi
