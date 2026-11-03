#!/bin/bash
# the footprint-share extraction over all cached validation frames (CPU, one core, nice 10; about 20 minutes)
cd "$(dirname "$0")/../.." || exit 1
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 nice -n 10 ~/miniconda3/envs/gcar/bin/python -u temp/coverage/footprint_share.py --val-only > temp/coverage/footprint_share.log 2>&1
echo "exit $?" >> temp/coverage/footprint_share.log
tail -3 temp/coverage/footprint_share.log
