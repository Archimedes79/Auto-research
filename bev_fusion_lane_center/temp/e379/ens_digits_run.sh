#!/bin/bash
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
for sp in val train; do nice -n 10 bash tools/env/_py14.sh temp/e379/ens_digits.py 1 $sp >> temp/e379/ens_digits.out 2>&1; done
