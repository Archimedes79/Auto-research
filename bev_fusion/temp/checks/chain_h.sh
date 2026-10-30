#!/bin/bash
# chain H: flow-explains with edge-aware smoothness (plain, then + sharp), after chain B
cd "$(dirname "$0")/../.."
sleep 60
while pgrep -f "[c]hain_b.sh" > /dev/null; do sleep 30; done
~/miniconda3/envs/gcar/bin/python -u tests/test_flow_explains.py --quick --smooth 0.05 > temp/checks/flow_explains_smooth_quick.log 2>&1
~/miniconda3/envs/gcar/bin/python -u tests/test_flow_explains.py --quick --smooth 0.05 --sharp > temp/checks/flow_explains_smooth_sharp_quick.log 2>&1
