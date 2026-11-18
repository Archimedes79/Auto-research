#!/bin/bash
# E377: n at its pinned (b) cell 0.2 / 0.2 / 0.6 / off, scored once on VAL against the champion at D115 and against m (E367 b5) at m's (b) cell (the same cell) (CPU, run.py)
cd "$(dirname "$0")/../.."
export BFLC_SIG=1.0
source temp/e377/texts377.sh
PY="env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nice -n 10 bash tools/env/_py14.sh"
CELL="--link-thr 0.2 --link-thr-long 0.2 --keep-thr 0.6 --no-none-gate"
for S in 0 1; do
  python3 ../autoresearch/tools/run.py --id E377.read_nb_s$S --seed $S --retries 0 --require-card --split val --timeout 90 --hyp "$HYP" --pred "$PRED" --rule "$RULE" -- bash -c "$PY jtoken_eval.py score --run e377_n_s$S --seed $S $CELL --ref-run e373_cb_s$S --ref-run-cell 0.3,0.2,0.6,off && $PY jtoken_eval.py score --run e377_n_s$S --seed $S $CELL --ref-run e367_b5_s$S" >> temp/e377/read_nb_s$S.out 2>&1
  echo "$(date '+%m-%d %H:%M:%S') [read nb_s$S] exit $?" >> temp/e377/chain377.log
done
