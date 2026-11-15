#!/bin/bash
# E291: the CPU variant of run_staged_gpu.sh for the tests: CUDA off (nothing may touch the GPU slots of the running jobs), 2 threads, nice, the cached PYTHONPATH unless E291_FAST=0.
#   wsl bash temp/e291/run_staged.sh <script.py | basename> [args ...]        BFLC_STAGE=none runs the real files (the control: tests of new rules must fail there)
D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 E291_FAST="${E291_FAST:-1}"
exec nice -n 10 bash "$D/run_staged_gpu.sh" "$@"
