#!/bin/bash
# chain D: the WTA arm of the flow test after chain C (same GPU slot)
cd "$(dirname "$0")/../.."
sleep 60
while pgrep -f "[c]hain_c.sh" > /dev/null; do sleep 30; done
~/miniconda3/envs/gcar/bin/python -u tests/test_flow_refine.py --quick --arms W,S > temp/checks/flow_refine_wta_quick.log 2>&1
