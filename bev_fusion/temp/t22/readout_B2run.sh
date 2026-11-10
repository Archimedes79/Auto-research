#!/bin/bash
# scratch: stage B2 read-out, CPU only, one thread per job, detached (setsid nohup) so it survives the end of the call
# gate rows (seconds), then the slow paired bootstrap B2 vs control and B2 vs stage B (about 13 min each, in parallel)
cd /mnt/c/Auto-research/bev_fusion || exit 1
export OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
PY=$HOME/miniconda3/envs/gcar/bin/python
L=temp/t22
$PY $L/chain/compare_gate.py runs/t21_pair4_s0 runs/t22_twin_pair4 runs/t22_B2_s0 --all > $L/gate_rows_B2.log 2>&1; echo "gate rows B2 exit $?"
setsid nohup $PY evaluation/bev_suite.py compare runs/t21_pair4_s0 runs/t22_B2_s0 --curves --fusion --velocity --n 1000 > $L/compare_B2_vs_control.log 2>&1 < /dev/null &
setsid nohup $PY evaluation/bev_suite.py compare runs/t22_B_s0 runs/t22_B2_s0 --curves --fusion --velocity --n 1000 > $L/compare_B2_vs_B.log 2>&1 < /dev/null &
sleep 3
ps -eo pid,etime,args | grep -E 'bev_suite' | grep -v grep | cut -c1-170
