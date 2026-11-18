#!/bin/bash
cd "$(dirname "$0")/../.."
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nice -n 10 bash tools/env/_py14.sh repro_score.py "$1" > temp/simplify/repro_$1.out 2>&1
echo "exit $?"
