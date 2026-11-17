#!/bin/bash
# E376 diagnostic (scratch): the empty-G measurement on both E373 heads, CPU, niced, 2 threads
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
for S in 0 1; do
  nice -n 10 bash tools/env/_py14.sh e376_emptyg.py e373_cb_s$S $S > temp/e376/emptyg_s$S.out 2>&1
  echo "seed $S exit $?"
done
