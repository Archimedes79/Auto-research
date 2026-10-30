#!/bin/bash
# chain T: best balance + edge-aware smoothness (the per-cell scatter inside boxes), 2 seeds, after chain S
cd "$(dirname "$0")/../.."
sleep 60
while pgrep -f "[c]hain_s.sh" > /dev/null; do sleep 30; done
PY=~/miniconda3/envs/gcar/bin/python
$PY -u tests/test_flow_explains.py --quick --sharp --pyr --doppler --rawwarp 3.0 --prior 0.03 --smooth 0.05 > temp/checks/flow_explains_lanes_best_smooth_s0.log 2>&1
$PY -u tests/test_flow_explains.py --quick --sharp --pyr --doppler --rawwarp 3.0 --prior 0.03 --smooth 0.05 --seed 1 > temp/checks/flow_explains_lanes_best_smooth_s1.log 2>&1
