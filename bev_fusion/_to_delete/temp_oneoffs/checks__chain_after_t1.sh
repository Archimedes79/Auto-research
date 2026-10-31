#!/bin/bash
# The nuScenes box tests and the recursive sequence evaluation - to start ONLY after the t1 rounds end (parent chat's
# reservation of GPU / CPU until about 13:00, 2026-10-26). GPU, one job at a time.
cd "$(dirname "$0")/../.."
PY=~/miniconda3/envs/gcar/bin/python
BEST="--sharp --pyr --doppler --rawwarp 3.0 --prior 0.03 --smooth 0.05"
$PY -u tests/test_flow_explains.py --quick --data nuscenes --recursive > temp/checks/flow_explains_nusc_baseline_s0.log 2>&1
$PY -u tests/test_flow_explains.py --quick --data nuscenes --recursive $BEST > temp/checks/flow_explains_nusc_best_s0.log 2>&1
$PY -u tests/test_flow_explains.py --data nuscenes --sequences --n-seq 24 --length 5 $BEST > temp/checks/flow_sequences_nusc_best_s0.log 2>&1
$PY -u tests/test_flow_explains.py --quick --data nuscenes --recursive $BEST --seed 1 > temp/checks/flow_explains_nusc_best_s1.log 2>&1
