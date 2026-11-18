#!/bin/bash
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
for s in 0 1; do nice -n 10 bash tools/env/_py14.sh temp/e379/gap_tokens.py $s > temp/e379/gap_tokens_s$s.out 2>&1; done
