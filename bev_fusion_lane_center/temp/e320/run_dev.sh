#!/bin/bash
# E317 scratch: the synthetic tests of the dev copies (CPU, 2 threads).  usage: run_dev.sh <selftest | regress>
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=
bash tools/env/_py14.sh e320/t_dev.py "$1"
