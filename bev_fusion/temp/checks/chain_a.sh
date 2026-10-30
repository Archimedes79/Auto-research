#!/bin/bash
# chain A (GPU slot 1 of mine): after the running S / S+R4 arms -> pyramid and prior arms -> full stage ladder
cd "$(dirname "$0")/../.."
# (no wait: the S / S+R4 run was stopped with its queue shell)
~/miniconda3/envs/gcar/bin/python -u tests/test_flow_refine.py --quick --arms S+R4,0+P4,S+P4,H+R4+prior > temp/checks/flow_refine_pyr_quick.log 2>&1
~/miniconda3/envs/gcar/bin/python -u tests/test_temporal_stages.py > temp/checks/temporal_stages_full.log 2>&1
