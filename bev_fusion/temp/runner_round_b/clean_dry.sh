#!/bin/bash
# scratch: dry run of clean_temp.py for the folder runner_round
cd /mnt/c/Auto-research || exit 1
AR_PROJECT=../bev_fusion python3 autoresearch/tools/clean_temp.py runner_round 2>&1 | tail -40 | cut -c1-230
echo "exit ${PIPESTATUS[0]}"
