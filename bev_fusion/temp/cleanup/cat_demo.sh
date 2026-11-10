#!/bin/bash
cd /mnt/c/Auto-research/autoresearch || exit 1
PY=$HOME/miniconda3/envs/gcar/bin/python
python3 -m jobrunner JobRunner.catalogue --json --project /mnt/c/Auto-research/bev_fusion --python $PY 2>&1 | python3 -c "
import json, sys
for r in json.load(sys.stdin):
    print(r['call'], '|', r['how'], '| needs', r['needs'], '|', r['parameters'][:150])"
