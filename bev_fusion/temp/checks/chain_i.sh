#!/bin/bash
# chain I: the supervised control H + R4 + label after chain F (same slot)
cd "$(dirname "$0")/../.."
sleep 240
while pgrep -f "[c]hain_f.sh" > /dev/null; do sleep 30; done
~/miniconda3/envs/gcar/bin/python -u tests/test_flow_refine.py --quick --arms H+R4+label > temp/checks/flow_refine_label_quick.log 2>&1
