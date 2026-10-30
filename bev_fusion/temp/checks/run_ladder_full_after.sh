#!/bin/bash
# start the full stage ladder once tests/test_flow_refine.py has finished (GPU slot limit)
cd "$(dirname "$0")/../.."
while pgrep -f "[t]est_flow_refine.py" > /dev/null; do sleep 20; done
~/miniconda3/envs/gcar/bin/python -u tests/test_temporal_stages.py > temp/checks/temporal_stages_full.log 2>&1
