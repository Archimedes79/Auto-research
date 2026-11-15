#!/bin/bash
cd /mnt/c/Auto-research/bev_fusion || exit 1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
date +%T
nice -n 10 bash tests/run_gates.sh 2>&1 | tee /mnt/c/Auto-research/bev_fusion/temp/cleanup/gates_main.log
date +%T
