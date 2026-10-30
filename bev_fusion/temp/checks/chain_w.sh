#!/bin/bash
# chain W: the nuScenes box tests on the CPU (GPU reserved for t1 until 13:00), one at a time, low priority
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES=
PY="nice -n 15 $HOME/miniconda3/envs/gcar/bin/python"
$PY -u tests/test_flow_explains.py --quick --data nuscenes --recursive > temp/checks/flow_explains_nusc_baseline_s0.log 2>&1
$PY -u tests/test_flow_explains.py --quick --data nuscenes --recursive --sharp --pyr --doppler --rawwarp 3.0 --prior 0.03 --smooth 0.05 > temp/checks/flow_explains_nusc_best_s0.log 2>&1
$PY -u tests/test_flow_explains.py --quick --data nuscenes --recursive --sharp --pyr --doppler --rawwarp 3.0 --prior 0.03 --smooth 0.05 --seed 1 > temp/checks/flow_explains_nusc_best_s1.log 2>&1
