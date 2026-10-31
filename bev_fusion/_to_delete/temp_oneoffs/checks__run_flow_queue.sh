#!/bin/bash
# my GPU queue (slot limit): after tests/test_flow_refine.py ends -> sharp arms S, S+R4 -> full stage ladder
cd "$(dirname "$0")/../.."
while pgrep -f "[t]est_flow_refine.py --quick$" > /dev/null; do sleep 20; done
~/miniconda3/envs/gcar/bin/python -u tests/test_flow_refine.py --quick --arms S,S+R4,0+P4,S+P4 > temp/checks/flow_refine_sharp_quick.log 2>&1
~/miniconda3/envs/gcar/bin/python -u tests/test_temporal_stages.py > temp/checks/temporal_stages_full.log 2>&1
