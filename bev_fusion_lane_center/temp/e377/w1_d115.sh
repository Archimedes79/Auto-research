#!/bin/bash
# E377 card measurement (diagnostic, CPU): the none-weight-1.0 heads (E371 cf, seed twins of the E373 heads) at the D115 cell, paired against the E373 heads at the same cell
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 BFLC_SIG=1.0
for S in 0 1; do
  nice -n 10 bash tools/env/_py14.sh jtoken_eval.py score --run e371_cf_s$S --seed $S --link-thr 0.3 --link-thr-long 0.2 --keep-thr 0.6 --no-none-gate --ref-run e373_cb_s$S > temp/e377/w1_d115_s$S.out 2>&1
  echo "seed $S exit $?"
done
