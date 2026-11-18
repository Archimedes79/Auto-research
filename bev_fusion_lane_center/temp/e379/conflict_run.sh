#!/bin/bash
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
for r in e373_cb_s0 e373_cb_s1; do nice -n 10 bash tools/env/_py14.sh temp/e379/conflict_long.py $r > temp/e379/conflict_$r.out 2>&1; done
