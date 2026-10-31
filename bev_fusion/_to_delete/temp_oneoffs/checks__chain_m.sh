#!/bin/bash
# chain M: box test, pyramid with / without the level leak on 2 seeds, and with per-iteration Doppler (after chain K)
cd "$(dirname "$0")/../.."
sleep 60
while pgrep -f "[c]hain_k.sh" > /dev/null; do sleep 30; done
PY=~/miniconda3/envs/gcar/bin/python
$PY -u tests/test_flow_explains.py --quick --sharp --pyr --pyr-leak > temp/checks/flow_explains_sharp_pyr_leak_s0.log 2>&1
$PY -u tests/test_flow_explains.py --quick --sharp --pyr --pyr-leak --seed 1 > temp/checks/flow_explains_sharp_pyr_leak_s1.log 2>&1
$PY -u tests/test_flow_explains.py --quick --sharp --pyr --seed 1 > temp/checks/flow_explains_sharp_pyr_s1.log 2>&1
$PY -u tests/test_flow_explains.py --quick --sharp --pyr --doppler > temp/checks/flow_explains_sharp_pyr_doppler_s0.log 2>&1
$PY -u tests/test_flow_explains.py --quick --sharp --pyr --doppler --seed 1 > temp/checks/flow_explains_sharp_pyr_doppler_s1.log 2>&1
