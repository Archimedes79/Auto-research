#!/usr/bin/env bash
cd "$(dirname "$0")/../.." || exit 1
env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMBA_NUM_THREADS=2 bash tools/env/_py14.sh experiments/E261_recall-weight-6-operating-point/frontier_scratch.py e123_D7_r3_rc1:288:e123_D7_r3_rc1,e18_D7_j1rcs1:192:e18_D7_j1rcs1 0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7
