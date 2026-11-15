#!/bin/bash
# scratch: three detached CPU jobs (one thread each) for the stage-B read-out; they survive the end of this call
cd /mnt/c/Auto-research/bev_fusion || exit 1
export OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
PY=$HOME/miniconda3/envs/gcar/bin/python
L=temp/t22
setsid nohup $PY evaluation/bev_suite.py compare runs/t21_pair4_s0 runs/t22_B_s0 --curves --fusion --n 1000 > $L/compare_B_vs_control_legacy.log 2>&1 < /dev/null &
setsid nohup $PY evaluation/bev_suite.py compare runs/t21_pair4_s0 runs/t22_B_s0 --curves --fusion --velocity --n 1000 > $L/compare_B_vs_control.log 2>&1 < /dev/null &
setsid nohup bash $L/compare_B2.sh > $L/compare_B2.out 2>&1 < /dev/null &
sleep 2
ps -eo pid,etime,args | grep -E 'bev_suite|compare_B2' | grep -v grep | cut -c1-170
