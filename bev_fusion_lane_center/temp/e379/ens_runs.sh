#!/bin/bash
# E379 arm e2 read-outs (CPU): the plain ensemble of the two teacher heads per seed at the D115 cell (jtoken_eval.py ensemble), registered by the card's run ids E379.ens_s0 / ens_s1
cd "$(dirname "$0")/../.."
export BFLC_SIG=1.0
source temp/e379/texts379.sh
for S in 0 1; do
  python3 ../autoresearch/tools/run.py --id E379.ens_s$S --seed $S --retries 0 --require-card --split val --timeout 30 --hyp "$HYP" --pred "$PRED" --rule "$RULE" -- env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nice -n 10 bash tools/env/_py14.sh jtoken_eval.py ensemble --runs e373_cb_s$S,e367_b25_s$S --seed $S --link-thr 0.3 --link-thr-long 0.2 --keep-thr 0.6 --no-none-gate --out-dir temp/e379/ens_cards >> temp/e379/ens_runs.out 2>&1
done
echo done >> temp/e379/ens_runs.out
