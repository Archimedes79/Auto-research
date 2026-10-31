#!/bin/bash
# chain K: flow-explains sharp + pyramid again after the pyramid detach fix (after chain H)
cd "$(dirname "$0")/../.."
sleep 120
while pgrep -f "[c]hain_h.sh" > /dev/null; do sleep 30; done
~/miniconda3/envs/gcar/bin/python -u tests/test_flow_explains.py --quick --sharp --pyr > temp/checks/flow_explains_sharp_pyr_fixed_quick.log 2>&1
~/miniconda3/envs/gcar/bin/python -u tests/test_flow_explains.py --quick --sharp --pyr --smooth 0.05 > temp/checks/flow_explains_sharp_pyr_smooth_quick.log 2>&1
