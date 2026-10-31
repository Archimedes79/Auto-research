#!/bin/bash
# chain B (GPU slot 2 of mine): flow-explains baseline -> flow-explains with sharp hypotheses + pyramid
cd "$(dirname "$0")/../.."
~/miniconda3/envs/gcar/bin/python -u tests/test_flow_explains.py --quick > temp/checks/flow_explains_quick.log 2>&1
~/miniconda3/envs/gcar/bin/python -u tests/test_flow_explains.py --quick --sharp --pyr > temp/checks/flow_explains_sharp_pyr_quick.log 2>&1
