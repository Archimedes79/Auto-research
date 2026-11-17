#!/bin/bash
# E377 step 'controls' (card v2): the matched-push control m = the E367 b5 heads: (a) at the D115 cell against the champion heads (cross-run interval), and their per-class TRAIN sweeps for m's (b) cell; CPU, through run.py
cd "$(dirname "$0")/../.."
export BFLC_SIG=1.0
source temp/e377/texts377.sh
PY="env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nice -n 10 bash tools/env/_py14.sh"
D115="--link-thr 0.3 --link-thr-long 0.2 --keep-thr 0.6 --no-none-gate"
for S in 0 1; do
  R=e367_b5_s$S
  python3 ../autoresearch/tools/run.py --id E377.ctl_m_s$S --seed $S --retries 0 --require-card --split val --timeout 150 --hyp "$HYP" --pred "$PRED" --rule "$RULE" -- bash -c "$PY jtoken_eval.py score --run $R --seed $S $D115 --ref-run e373_cb_s$S && $PY jtoken_eval.py sweep --run $R --seed $S --split train --every 3 --grid perclass" >> temp/e377/ctl_m_s$S.out 2>&1
  echo "$(date '+%m-%d %H:%M:%S') [ctl m_s$S] exit $?" >> temp/e377/chain377.log
done
