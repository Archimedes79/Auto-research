#!/bin/bash
# scratch: test_lane_live.py selftests (the selftest and the real-data commands, one child at a time), CPU only
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=
nice -n 10 bash tools/env/_py14.sh training/grid/test_lane_live.py selftests > temp/e372/e372t_lanelive.txt 2>&1
echo "EXIT $?" >> temp/e372/e372t_lanelive.txt
tail -12 temp/e372/e372t_lanelive.txt
