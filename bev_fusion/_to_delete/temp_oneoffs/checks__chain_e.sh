#!/bin/bash
# chain E: a stronger zero-motion prior (0.03) after chain D (same GPU slot)
cd "$(dirname "$0")/../.."
sleep 120
while pgrep -f "[c]hain_d.sh" > /dev/null; do sleep 30; done
~/miniconda3/envs/gcar/bin/python -u tests/test_flow_refine.py --quick --arms H+R4+prior --prior 0.03 > temp/checks/flow_refine_prior003_quick.log 2>&1
