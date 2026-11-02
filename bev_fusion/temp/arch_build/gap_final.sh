#!/usr/bin/env bash
# one-off: the checks after the critic's fixes (CPU, 2 threads): precompute check on real frames incl. the fan against the GT layer,
# dataset self-test (float gap steps), test_slots, train.py self-test (the batch's gap steps against the GT's)
cd "$(dirname "$0")/../.." || exit 1
export OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=
P=~/miniconda3/envs/gcar/bin/python
echo "=== precompute check"; $P training/precompute.py check training/configs/t10_dual_s0.json --n 4 2>&1 | grep -v Warning | tail -3
echo "=== nuscenes_dataset"; $P training/nuscenes_dataset.py 2>&1 | grep -v Warning | tail -3
echo "=== test_slots"; $P tests/test_slots.py 2>&1 | grep -v Warning | tail -3
echo "=== train selftest"; $P training/train.py 2>&1 | grep -v Warning | tail -5
echo "=== done"
