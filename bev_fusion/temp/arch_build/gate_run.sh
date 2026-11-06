#!/usr/bin/env bash
# one-off: selftests after the Doppler gate (CPU, 2 threads)
cd "$(dirname "$0")/../.." || exit 1
export OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=
P=~/miniconda3/envs/gcar/bin/python
echo "=== precompute"; $P training/precompute.py 2>&1 | grep -v Warning | tail -3
echo "=== augment"; $P training/augment.py 2>&1 | grep -v Warning | tail -2 | cut -c1-200
echo "=== test_slots"; $P tests/test_slots.py 2>&1 | grep -v Warning | tail -6
echo "=== bundle"; $P tests/test_bundle.py 2>&1 | tail -2
echo "=== done"
