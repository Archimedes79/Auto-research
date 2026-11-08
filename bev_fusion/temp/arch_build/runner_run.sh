#!/usr/bin/env bash
# one-off: the runner test on the real run (CPU, 2 threads)
cd "$(dirname "$0")/../.." || exit 1
export OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=
~/miniconda3/envs/gcar/bin/python tests/test_runner.py 2>&1 | grep -v Warning | tail -25
echo "=== done"
