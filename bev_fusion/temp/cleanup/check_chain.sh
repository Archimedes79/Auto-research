#!/bin/bash
# look-only check (no submit) of the four example chain job files
cd /mnt/c/Auto-research/autoresearch || exit 1
PY=$HOME/miniconda3/envs/gcar/bin/python
export AR_PROJECT=../bev_fusion
for j in gt train predict score; do
  echo "=== $j"
  python3 -m jobrunner JobRunner.check ../bev_fusion/records/jobs/chain/$j.json --project ../bev_fusion --python $PY 2>&1 | head -40
  echo "exit=$?"
done
