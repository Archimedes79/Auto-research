#!/bin/bash
# chain Q: box test on the corrected data (lanes present), best setup 2 seeds + baseline
cd "$(dirname "$0")/../.."
PY=~/miniconda3/envs/gcar/bin/python
$PY -u tests/test_flow_explains.py --quick --sharp --pyr --doppler --rawwarp 1.0 > temp/checks/flow_explains_lanes_best_s0.log 2>&1
$PY -u tests/test_flow_explains.py --quick --sharp --pyr --doppler --rawwarp 1.0 --seed 1 > temp/checks/flow_explains_lanes_best_s1.log 2>&1
$PY -u tests/test_flow_explains.py --quick > temp/checks/flow_explains_lanes_baseline_s0.log 2>&1
