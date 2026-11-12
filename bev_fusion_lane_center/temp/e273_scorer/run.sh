#!/bin/bash
# run a scratch script of temp/e273_scorer on the CPU only: run.sh <script.py> [args]
cd "$(dirname "$0")/../.."
S="$1"; shift
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMBA_NUM_THREADS=2
exec nice -n 12 bash tools/env/_py14.sh temp/e273_scorer/$S "$@"
