#!/bin/bash
# scratch: the folder's gate runner (CPU) detached at low priority; the output goes to temp/t22/run_gates.out
cd /mnt/c/Auto-research/bev_fusion || exit 1
export OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=""
setsid nohup nice -n 12 bash tests/run_gates.sh > temp/t22/run_gates.out 2>&1 < /dev/null &
sleep 3
ps -eo pid,etime,args | grep -E 'run_gates' | grep -v grep | cut -c1-120
