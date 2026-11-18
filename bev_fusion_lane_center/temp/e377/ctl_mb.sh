#!/bin/bash
# E377: control m (E367 b5 heads) at its pinned (b) cell 0.2 / 0.2 / 0.6 / off, scored once on VAL against the champion heads at the D115 cell (CPU, through run.py)
cd "$(dirname "$0")/../.."
export BFLC_SIG=1.0
source temp/e377/texts377.sh
PY="env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nice -n 10 bash tools/env/_py14.sh"
for S in 0 1; do
  python3 ../autoresearch/tools/run.py --id E377.ctl_mb_s$S --seed $S --retries 0 --require-card --split val --timeout 60 --hyp "$HYP" --pred "$PRED" --rule "$RULE" -- bash -c "$PY jtoken_eval.py score --run e367_b5_s$S --seed $S --link-thr 0.2 --link-thr-long 0.2 --keep-thr 0.6 --no-none-gate --ref-run e373_cb_s$S --ref-run-cell 0.3,0.2,0.6,off" >> temp/e377/ctl_mb_s$S.out 2>&1
  echo "$(date '+%m-%d %H:%M:%S') [ctl mb_s$S] exit $?" >> temp/e377/chain377.log
done
