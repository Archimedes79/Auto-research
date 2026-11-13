#!/bin/bash
# E320 scratch: py_compile + self-tests of the DEV read-out modules (qends_eval, jmaneuver, jends_eval) through the loaders of t_real.py (no second *.py copies under temp/)
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=
bash tools/env/_py14.sh e320/t_real.py selftest_dev 2>&1 | grep -v "^_py14" | tail -n 12
