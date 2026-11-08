#!/usr/bin/env bash
# one-off: the runner test on the real run (CPU, 2 threads), raw output and the exit code in the log
cd "$(dirname "$0")/../.." || exit 1
export OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=
~/miniconda3/envs/gcar/bin/python tests/test_runner.py > temp/arch_build/runner_raw.log 2>&1
echo "exit code $?" >> temp/arch_build/runner_raw.log
echo "=== done"
