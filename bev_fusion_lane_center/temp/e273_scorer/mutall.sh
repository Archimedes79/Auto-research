#!/bin/bash
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMBA_NUM_THREADS=2
for m in "$@"; do
  set -- ${m//:/ }
  nice -n 12 bash tools/env/_py14.sh temp/e273_scorer/mutrun.py "$@" 2>&1 | grep -E "^(MUTANT|FAIL|ok |Traceback|.*Error)"
done
echo ALLDONE
