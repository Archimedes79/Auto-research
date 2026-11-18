#!/bin/bash
# the floor probe of the ensemble (temp/e379/e379e_floor_probe.py) on both seeds
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
for s in 0 1; do nice -n 10 bash tools/env/_py14.sh temp/e379/e379e_floor_probe.py $s; done > temp/e379/e379e_floor_probe.out 2>&1
