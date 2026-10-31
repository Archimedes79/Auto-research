#!/bin/bash
# chain N: seed 2 of the box test, pyramid with and without the level leak (inconclusive on 2 seeds), after chain M
cd "$(dirname "$0")/../.."
sleep 60
while pgrep -f "[c]hain_m.sh" > /dev/null; do sleep 30; done
PY=~/miniconda3/envs/gcar/bin/python
$PY -u tests/test_flow_explains.py --quick --sharp --pyr --pyr-leak --seed 2 > temp/checks/flow_explains_sharp_pyr_leak_s2.log 2>&1
$PY -u tests/test_flow_explains.py --quick --sharp --pyr --seed 2 > temp/checks/flow_explains_sharp_pyr_s2.log 2>&1
