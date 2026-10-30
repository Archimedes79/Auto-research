#!/bin/bash
# chain U (2nd version): nuScenes box test with the recursive evaluation - baseline, then the best setup (seed 0)
cd "$(dirname "$0")/../.."
PY=~/miniconda3/envs/gcar/bin/python
$PY -u tests/test_flow_explains.py --quick --data nuscenes --recursive > temp/checks/flow_explains_nusc_baseline_s0.log 2>&1
$PY -u tests/test_flow_explains.py --quick --data nuscenes --recursive --sharp --pyr --doppler --rawwarp 3.0 --prior 0.03 --smooth 0.05 > temp/checks/flow_explains_nusc_best_s0.log 2>&1
