#!/bin/bash
# the fast gates of bev_fusion (tests/run_gates.sh without --all) next to the train.py selftest: CPU only, nice 10, 2 threads
cd "$(dirname "$0")/../.." || exit 1
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 nice -n 10 bash tests/run_gates.sh > temp/pitch/gates_fast.log 2>&1
echo "gates exit $?" >> temp/pitch/gates_fast.log
cat temp/pitch/gates_fast.log | cut -c1-200
