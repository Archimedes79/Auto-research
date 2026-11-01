#!/usr/bin/env bash
# E261.cv (loop 22, CPU): rc2 / rc1 operating points chosen out-of-fold on 20 scene halves (card section 2, script e261_cv.py)
cd "$(dirname "$0")/../.." || exit 1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMBA_NUM_THREADS=2
bash tools/env/_py14.sh temp/loop22/e261_cv.py
