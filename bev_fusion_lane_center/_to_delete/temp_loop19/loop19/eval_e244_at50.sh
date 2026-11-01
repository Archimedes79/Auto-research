#!/bin/bash
# E244 partial reading: the 50 % checkpoint of arm a seed 0 on VAL 873 (evaluate.py), after e244_prep_at50.py
cd "$(dirname "$0")/../.."
python3 temp/loop19/e244_prep_at50.py
bash tools/env/_py14.sh evaluate.py e24_j1phd_s0_at50 2>&1 | grep -E "^  S|on val|Error|Traceback" | cut -c1-230
