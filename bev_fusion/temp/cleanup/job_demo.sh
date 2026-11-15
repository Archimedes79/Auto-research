#!/bin/bash
# read-only: check a demo job and list the catalogue against the real BEV project, the interpreter given as an option; nothing is submitted or written
cd /mnt/c/Auto-research/autoresearch || exit 1
PY=$HOME/miniconda3/envs/gcar/bin/python
echo "== check, no option (system python: no torch)"
( time python3 -m jobrunner JobRunner.check /mnt/c/Auto-research/bev_fusion/temp/cleanup/job_demo.json --project /mnt/c/Auto-research/bev_fusion ) 2>&1 | grep -E "^ok:|warnings|not checked|real|problems" | cut -c1-200
echo "== check with --python gcar"
( time python3 -m jobrunner JobRunner.check /mnt/c/Auto-research/bev_fusion/temp/cleanup/job_demo.json --project /mnt/c/Auto-research/bev_fusion --python $PY ) 2>&1 | grep -E "^ok:|warnings|not checked|real|problems|^  - " | cut -c1-200 | head -14
echo "== catalogue with --python gcar"
python3 -m jobrunner JobRunner.catalogue --project /mnt/c/Auto-research/bev_fusion --python $PY 2>&1 | grep -E "call:|parameters:" | cut -c1-220
echo "== files written under the BEV records (must be none)"
ls -d /mnt/c/Auto-research/bev_fusion/records/runs/.jobs 2>&1 | head -2
