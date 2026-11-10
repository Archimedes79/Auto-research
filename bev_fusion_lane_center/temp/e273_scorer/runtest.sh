#!/bin/bash
# the scorer tests on the CPU, 2 threads, low priority: runtest.sh [word ...]
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMBA_NUM_THREADS=2
exec nice -n 12 bash tools/env/_py14.sh test_qgraph_eval.py "$@"
