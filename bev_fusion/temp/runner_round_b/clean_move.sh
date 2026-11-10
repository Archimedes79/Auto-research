#!/bin/bash
# scratch: tag my scratch folder (runner_round) and move it to _to_delete (the 6-hour guard only is forced: the jobs of the folder are finished)
cd /mnt/c/Auto-research || exit 1
AR_PROJECT=../bev_fusion python3 autoresearch/tools/clean_temp.py runner_round --move --force 2>&1 | tail -20 | cut -c1-230
echo "exit ${PIPESTATUS[0]}"
git tag -l "bf/runner_round*" | cat
ls bev_fusion/_to_delete | head
