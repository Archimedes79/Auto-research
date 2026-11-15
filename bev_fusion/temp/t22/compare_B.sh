#!/bin/bash
# scratch: stage B of the t22 card after its evaluation: paired scene bootstrap of t22_B_s0 against the control t21_pair4_s0 (the card's control) and against the live-code twin, and the gate rows as point estimates; CPU only
cd /mnt/c/Auto-research/bev_fusion || exit 1
export OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=""
PY=$HOME/miniconda3/envs/gcar/bin/python
L=temp/t22
test -f runs/t22_B_s0/eval/card.json || { echo "no card.json of runs/t22_B_s0: the evaluation did not finish"; exit 1; }
$PY evaluation/bev_suite.py compare runs/t21_pair4_s0 runs/t22_B_s0 --curves --fusion --velocity --n 1000 > $L/compare_B_vs_control.log 2>&1; echo "compare vs control exit $?"
$PY evaluation/bev_suite.py compare runs/t22_twin_pair4 runs/t22_B_s0 --curves --fusion --velocity --n 1000 > $L/compare_B_vs_twin.log 2>&1; echo "compare vs twin exit $?"
$PY evaluation/bev_suite.py compare runs/t22_standin_pair4 runs/t22_B_s0 --curves --fusion --velocity --n 1000 > $L/compare_B_vs_standin.log 2>&1; echo "compare vs stand-in exit $?"
$PY $L/chain/compare_gate.py runs/t21_pair4_s0 runs/t22_twin_pair4 runs/t22_B_s0 --all > $L/gate_rows_B.log 2>&1; echo "gate rows exit $?"
$PY $L/chain/compare_gate.py runs/t21_pair4_s0 runs/t22_twin_pair4 runs/t22_standin_pair4 > $L/gate_rows_standin.log 2>&1; echo "gate rows stand-in exit $?"
