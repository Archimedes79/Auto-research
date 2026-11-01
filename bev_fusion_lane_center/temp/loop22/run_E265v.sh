#!/usr/bin/env bash
cd "$(dirname "$0")/../.." || exit 1
env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMBA_NUM_THREADS=2 bash tools/env/_py14.sh experiments/E265_far-coverage-loss-budget/cleanup_split.py recall_s1 e24_j1rc_s1 e24_j1rc_s2 e24_j1rc_s3 champion fast_recall_na
