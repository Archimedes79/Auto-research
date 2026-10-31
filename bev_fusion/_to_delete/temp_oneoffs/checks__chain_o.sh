#!/bin/bash
# chain O: box test, sharp + pyramid + per-iteration Doppler + raw-warp loss, 2 seeds (parallel to chain N)
cd "$(dirname "$0")/../.."
PY=~/miniconda3/envs/gcar/bin/python
$PY -u tests/test_flow_explains.py --quick --sharp --pyr --doppler --rawwarp 1.0 > temp/checks/flow_explains_doppler_raw_s0.log 2>&1
$PY -u tests/test_flow_explains.py --quick --sharp --pyr --doppler --rawwarp 1.0 --seed 1 > temp/checks/flow_explains_doppler_raw_s1.log 2>&1
