#!/bin/bash
# scratch: the rest of compare_B.sh run in parallel with the (slow) bootstrap against the control: B against the frozen stand-in, and the gate-row tables; CPU only, one thread each
cd /mnt/c/Auto-research/bev_fusion || exit 1
export OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
PY=$HOME/miniconda3/envs/gcar/bin/python
L=temp/t22
$PY $L/chain/compare_gate.py runs/t21_pair4_s0 runs/t22_twin_pair4 runs/t22_B_s0 --all > $L/gate_rows_B.log 2>&1; echo "gate rows B exit $?"
$PY $L/chain/compare_gate.py runs/t21_pair4_s0 runs/t22_twin_pair4 runs/t22_standin_pair4 > $L/gate_rows_standin.log 2>&1; echo "gate rows stand-in exit $?"
$PY evaluation/bev_suite.py compare runs/t22_standin_pair4 runs/t22_B_s0 --curves --fusion --velocity --n 1000 > $L/compare_B_vs_standin.log 2>&1; echo "compare vs stand-in exit $?"
