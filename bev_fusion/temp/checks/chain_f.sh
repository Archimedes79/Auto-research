#!/bin/bash
# chain F: the edge-aware smoothness arm after chain E (same GPU slot)
cd "$(dirname "$0")/../.."
sleep 180
while pgrep -f "[c]hain_e.sh" > /dev/null; do sleep 30; done
~/miniconda3/envs/gcar/bin/python -u tests/test_flow_refine.py --quick --arms S+R4+smooth > temp/checks/flow_refine_smooth_quick.log 2>&1
