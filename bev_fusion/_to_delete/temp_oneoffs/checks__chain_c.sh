#!/bin/bash
# chain C: Doppler arms of the flow test, once one of my GPU slots frees (camera-only test or chain B done)
cd "$(dirname "$0")/../.."
while pgrep -f "[t]est_camera_only.py" > /dev/null && pgrep -f "[c]hain_b.sh" > /dev/null; do sleep 30; done
~/miniconda3/envs/gcar/bin/python -u tests/test_flow_refine.py --quick --arms S+D,S+D+P4 > temp/checks/flow_refine_doppler_quick.log 2>&1
