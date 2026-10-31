#!/bin/bash
# chain P: the best box-test configuration at the full budget (1500 sequences, 3000 steps), after chain O
cd "$(dirname "$0")/../.."
sleep 60
while pgrep -f "[c]hain_o.sh" > /dev/null; do sleep 30; done
~/miniconda3/envs/gcar/bin/python -u tests/test_flow_explains.py --sharp --pyr --doppler --rawwarp 1.0 > temp/checks/flow_explains_best_full_s0.log 2>&1
