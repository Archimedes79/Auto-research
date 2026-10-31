#!/bin/bash
# chain J: the pyramid arms again after the detach fix (after chain I, same slot)
cd "$(dirname "$0")/../.."
sleep 300
while pgrep -f "[c]hain_i.sh" > /dev/null; do sleep 30; done
~/miniconda3/envs/gcar/bin/python -u tests/test_flow_refine.py --quick --arms 0+P4,S+P4,S+D+P4 > temp/checks/flow_refine_pyr_fixed_quick.log 2>&1
