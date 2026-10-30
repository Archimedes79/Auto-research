#!/bin/bash
# chain R: box test on the corrected data - zero-motion prior 0.03 (lane aperture), stronger raw-warp (decoder share); after chain Q
cd "$(dirname "$0")/../.."
sleep 60
while pgrep -f "[c]hain_q.sh" > /dev/null; do sleep 30; done
PY=~/miniconda3/envs/gcar/bin/python
$PY -u tests/test_flow_explains.py --quick --sharp --pyr --doppler --rawwarp 1.0 --prior 0.03 > temp/checks/flow_explains_lanes_prior003_s0.log 2>&1
$PY -u tests/test_flow_explains.py --quick --sharp --pyr --doppler --rawwarp 3.0 > temp/checks/flow_explains_lanes_raw3_s0.log 2>&1
