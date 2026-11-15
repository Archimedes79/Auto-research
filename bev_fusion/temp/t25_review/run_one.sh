#!/bin/bash
# usage: run_one.sh <logname> <script> [args]
cd /mnt/c/Auto-research/bev_fusion
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
log=temp/t25_review/$1.log; shift
nice -n 15 ~/miniconda3/envs/gcar/bin/python "$@" > $log 2>&1
echo "exit $?" >> $log
