#!/bin/bash
# read-only: does the catalogue of the BEV project now contain the GT calls from GtGenerator.JOBS?
cd /mnt/c/Auto-research/autoresearch || exit 1
python3 -m jobrunner JobRunner.catalogue --json --project /mnt/c/Auto-research/bev_fusion 2>&1 | python3 -c "
import json, sys
for r in json.load(sys.stdin):
    print(r['call'], '|', r['how'], '| needs', r['needs'], '| reads', r['reads'], '| writes', r['writes'])" | cut -c1-230
ls -d /mnt/c/Auto-research/bev_fusion/records/runs/.jobs 2>&1 | head -1
