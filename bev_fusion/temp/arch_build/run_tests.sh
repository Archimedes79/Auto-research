#!/usr/bin/env bash
# one-off: the self-tests and the slot test of the architecture build (CPU)
cd "$(dirname "$0")/../.." || exit 1
export OMP_NUM_THREADS=4
P=~/miniconda3/envs/gcar/bin/python
for t in training/precompute.py training/interface.py training/extract.py tests/test_slots.py; do
  $P $t 2>&1 | tail -1
done
