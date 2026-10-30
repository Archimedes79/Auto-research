#!/bin/bash
# chain S: raw-warp 3 + prior 0.03 (vehicles + lanes), 2 seeds; raw-warp 3 seed 1
cd "$(dirname "$0")/../.."
PY=~/miniconda3/envs/gcar/bin/python
$PY -u tests/test_flow_explains.py --quick --sharp --pyr --doppler --rawwarp 3.0 --prior 0.03 > temp/checks/flow_explains_lanes_raw3_prior_s0.log 2>&1
$PY -u tests/test_flow_explains.py --quick --sharp --pyr --doppler --rawwarp 3.0 --prior 0.03 --seed 1 > temp/checks/flow_explains_lanes_raw3_prior_s1.log 2>&1
$PY -u tests/test_flow_explains.py --quick --sharp --pyr --doppler --rawwarp 3.0 --seed 1 > temp/checks/flow_explains_lanes_raw3_s1.log 2>&1
