#!/bin/bash
# chain L: Doppler after every refinement iteration, then the pyramid arms after the detach fix (after chain I)
cd "$(dirname "$0")/../.."
while pgrep -f "[c]hain_i.sh" > /dev/null; do sleep 30; done
~/miniconda3/envs/gcar/bin/python -u tests/test_flow_refine.py --quick --arms S+DD+R4+smooth > temp/checks/flow_refine_dd_quick.log 2>&1
~/miniconda3/envs/gcar/bin/python -u tests/test_flow_refine.py --quick --arms 0+P4,S+P4,S+D+P4 > temp/checks/flow_refine_pyr_fixed_quick.log 2>&1
